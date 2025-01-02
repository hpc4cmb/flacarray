# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Common I/O functions.

This module contains helper tools used by multiple I/O formats.

"""

import logging

import numpy as np

from .utils import keep_select, function_timer, select_keep_indices


log = logging.getLogger("flacarray")


@function_timer
def read_compressed_dataset_slice(dcomp, keep, stream_starts, stream_nbytes):
    """Read compressed bytes directly from an open dataset.

    This function works with zarr or h5py datasets.

    The `keep` and `stream_starts` are relative to the full dataset (i.e. they are
    "global", not local to a process if using MPI).

    Args:
        dcomp (Dataset):  The open dataset with compressed bytes.
        keep (array):  Bool array of streams to keep in the decompression.
        stream_starts (array):  The array of starting bytes in the dataset.
        stream_nbytes (array):  The array of number of bytes in the dataset.

    Returns:
        (tuple):  The (loaded data, rel_starts, indices).

    """
    if keep is None:
        # Load the full, contiguous bytes for all streams
        total_bytes = np.sum(stream_nbytes)
        if total_bytes == 0:
            return (None, None, None)
        start_byte = stream_starts.flatten()[0]
        rel_starts = stream_starts - start_byte
        dslc = (slice(0, total_bytes),)
        hslc = (slice(start_byte, start_byte + total_bytes),)
        data = np.empty(total_bytes, dtype=np.uint8)
        if hasattr(dcomp, "read_direct"):
            # HDF5
            dcomp.read_direct(data, hslc, dslc)
        else:
            # Zarr
            data[dslc] = dcomp[hslc]
        return (data, rel_starts, None)
    else:
        # We are reading a subset of streams.  Preallocate the read buffer and then
        # do multiple reads to fill sections of that buffer.
        starts, nbytes, indices = keep_select(keep, stream_starts, stream_nbytes)
        if len(starts) == 0:
            return (None, None, None)
        total_bytes = np.sum(nbytes)
        rel_starts = np.zeros_like(starts)
        rel_starts[1:] = np.cumsum(nbytes)[:-1]
        data = np.empty(total_bytes, dtype=np.uint8)
        if hasattr(dcomp, "read_direct"):
            # HDF5
            for istr in range(len(starts)):
                dslc = (slice(rel_starts[istr], rel_starts[istr] + nbytes[istr]),)
                hslc = (slice(starts[istr], starts[istr] + nbytes[istr]),)
                dcomp.read_direct(data, hslc, dslc)
        else:
            # Zarr
            for istr in range(len(starts)):
                dslc = (slice(rel_starts[istr], rel_starts[istr] + nbytes[istr]),)
                hslc = (slice(starts[istr], starts[istr] + nbytes[istr]),)
                data[dslc] = dcomp[hslc]
        return (data, rel_starts, indices)


def check_requests(reqs, wait_all=False):
    """Check the status of MPI send requests.

    If a send has completed, delete the associated send buffer, otherwise continue.
    If wait_all is True, wait for all current requests to finish.

    Args:
        reqs (dict):  Dictionary of per process send buffers.
        wait_all (bool):  If True, wait for all requests.

    Returns:
        None

    """
    procs = list(reqs.keys())
    for p in procs:
        preq = reqs[p]
        tags = list(preq.keys())
        for t in tags:
            rq, buf = preq[t]
            if wait_all:
                rq.wait()
                del buf
                del preq[t]
            else:
                if rq.test():
                    # Completed
                    del buf
                    del preq[t]


@function_timer
def read_send_compressed(reader, global_shape, keep=None, mpi_comm=None, mpi_dist=None):
    """Read data on one process and distribute.

    Args:


    Returns:
        (tuple):  The data and metadata

    """
    if mpi_comm is None:
        nproc = 1
        rank = 0
    else:
        nproc = mpi_comm.size
        rank = mpi_comm.rank

    global_leading_shape = global_shape[:-1]
    stream_size = global_shape[-1]

    local_shape = None
    local_starts = None
    stream_nbytes = None
    compressed = None
    stream_offsets = None
    stream_gains = None
    keep_indices = None

    # One process reads and sends.
    # The rank zero process will read data and send to the other
    # processes.  Keep a handle to the asynchronous send buffers
    # and delete them after the sends are complete.
    requests = dict()
    for proc in range(nproc):
        # While reading the per-process chunks, free any send buffers
        # that have completed.
        check_requests(requests)

        if rank == 0:
            # The range of the leading dimension on this process.
            send_range = mpi_dist[proc]
            send_leading_shape = (
                send_range[1] - send_range[0],
            ) + global_leading_shape[1:]

            # The helper datasets all have the same slab definitions
            dslc = tuple([slice(0, x) for x in send_leading_shape])
            fslc = (
                slice(send_range[0], send_range[0] + send_leading_shape[0]),
            ) + tuple([slice(0, x) for x in send_leading_shape[1:]])

            # If we are using the "keep" array to select streams, slice that
            # to cover only data for this process.
            if keep is None:
                proc_keep = None
            else:
                proc_keep = keep[dslc]

            # Stream starts
            raw_starts = reader.load_starts(mpi_comm, fslc, dslc)

            # Stream nbytes
            raw_nbytes = reader.load_nbytes(mpi_comm, fslc, dslc)

            # Offsets and gains for type conversions
            raw_offsets = reader.load_offsets(mpi_comm, fslc, dslc)
            raw_gains = reader.load_gains(mpi_comm, fslc, dslc)

            # Compressed bytes.  Apply our stream selection and load just those
            # streams we are keeping for this process.
            dcomp = reader.compressed_dataset
            proc_compressed, proc_starts, proc_keep_indices = (
                read_compressed_dataset_slice(dcomp, proc_keep, raw_starts, raw_nbytes)
            )

            # Cut our other arrays to only include the indices selected by the keep
            # mask.
            proc_nbytes = select_keep_indices(raw_nbytes, proc_keep_indices)
            proc_offsets = select_keep_indices(raw_offsets, proc_keep_indices)
            proc_gains = select_keep_indices(raw_gains, proc_keep_indices)

            if proc_starts is None:
                # This rank has no data after masking
                proc_shape = None
            else:
                proc_shape = tuple(proc_starts.shape)

            if proc == 0:
                # Store local data
                if proc_shape is not None:
                    local_shape = proc_shape + (stream_size,)
                local_starts = proc_starts
                stream_nbytes = proc_nbytes
                stream_offsets = proc_offsets
                stream_gains = proc_gains
                compressed = proc_compressed
                keep_indices = proc_keep_indices
            else:
                # Send to correct process.
                buffers = [proc_starts, proc_nbytes, proc_compressed]
                if proc_offsets is not None:
                    buffers.append(proc_offsets)
                if proc_gains is not None:
                    buffers.append(proc_gains)

                # Send two pieces of information needed to receiver further data.
                requests[proc] = dict()
                max_n_send = 7
                tag_base = max_n_send * proc

                req = mpi_comm.isend(proc_shape, dest=proc, tag=tag_base)
                requests[proc][0] = (req, proc_shape)

                req = mpi_comm.isend(proc_keep_indices, dest=proc, tag=tag_base + 1)
                requests[proc][0] = (req, proc_keep_indices)

                if proc_shape is not None:
                    # This process has some data
                    for itag, buf in enumerate(buffers):
                        req = mpi_comm.Isend(buf, dest=proc, tag=tag_base + itag + 2)
                        requests[proc][itag] = (req, buf)
        elif proc == rank:
            # First receive the shape and keep indices, which may change depending on
            # keep mask.
            max_n_recv = 7
            tag_base = max_n_recv * proc
            proc_shape = mpi_comm.recv(source=0, tag=tag_base)
            proc_keep_indices = mpi_comm.recv(source=0, tag=tag_base + 1)
            if proc_shape is not None:
                # This process has some data
                local_shape = proc_shape + (stream_size,)
                keep_indices = proc_keep_indices

                local_starts = np.empty(proc_shape, dtype=np.int64)
                mpi_comm.Recv(local_starts, source=0, tag=tag_base + 2)

                stream_nbytes = np.empty(proc_shape, dtype=np.int64)
                mpi_comm.Recv(stream_nbytes, source=0, tag=tag_base + 3)

                total_bytes = np.sum(stream_nbytes)
                compressed = np.empty(total_bytes, dtype=np.uint8)
                mpi_comm.Recv(compressed, source=0, tag=tag_base + 4)

                if reader.stream_off_dtype is not None:
                    stream_offsets = np.empty(proc_shape, dtype=reader.stream_off_dtype)
                    mpi_comm.Recv(stream_offsets, source=0, tag=tag_base + 5)

                if reader.stream_gain_dtype is not None:
                    stream_gains = np.empty(proc_shape, dtype=reader.stream_gain_dtype)
                    mpi_comm.Recv(stream_gains, source=0, tag=tag_base + 6)
    check_requests(requests, wait_all=True)

    return (
        local_shape,
        local_starts,
        stream_nbytes,
        compressed,
        stream_offsets,
        stream_gains,
        keep_indices,
    )


@function_timer
def receive_write_compressed(
    writer, global_leading_shape, global_process_nbytes, mpi_comm=None, mpi_dist=None
):
    if mpi_comm is None:
        nproc = 1
        rank = 0
    else:
        nproc = mpi_comm.size
        rank = mpi_comm.rank

    # Compute the byte offset of each process's data
    comp_doff = list()
    coff = 0
    for proc in range(nproc):
        comp_doff.append(coff)
        coff += global_process_nbytes[proc]

    for proc in range(nproc):
        # Set up communication tags for the buffers we will send / receive
        tag_nbuf = 5
        tag_comp = tag_nbuf * proc + 0
        tag_starts = tag_nbuf * proc + 1
        tag_nbytes = tag_nbuf * proc + 2
        tag_stream_offsets = tag_nbuf * proc + 3
        tag_stream_gains = tag_nbuf * proc + 4
        if rank == 0:
            # The rank zero process will receive data from the other processes
            # and write it into the global datasets.  For each dataset we build
            # the "slab" (tuple of slices) that we will write from the array
            # in memory and to the HDF5 dataset.
            #
            # The range of the leading dimension on this process.
            recv_range = mpi_dist[proc]
            recv_leading_shape = (
                recv_range[1] - recv_range[0],
            ) + global_leading_shape[1:]

            # The next 4 datasets all have the same slab definitions
            dslc = tuple([slice(0, x) for x in recv_leading_shape])
            fslc = (
                slice(recv_range[0], recv_range[0] + recv_leading_shape[0]),
            ) + tuple([slice(0, x) for x in recv_leading_shape[1:]])

            # Stream starts
            if proc == 0:
                recv = writer.starts.astype(np.int64)
            else:
                recv = np.empty(recv_leading_shape, dtype=np.int64)
                mpi_comm.Recv(recv, source=proc, tag=tag_starts)
            writer.save_starts(recv, mpi_comm, dslc, fslc)
            del recv

            # Stream nbytes
            if proc == 0:
                recv = writer.nbytes.astype(np.int64)
            else:
                recv = np.empty(recv_leading_shape, dtype=np.int64)
                mpi_comm.Recv(recv, source=proc, tag=tag_nbytes)
            writer.save_nbytes(recv, mpi_comm, dslc, fslc)
            del recv

            # Offsets and gains for type conversions
            if writer.have_offsets:
                if proc == 0:
                    recv = writer.offsets
                else:
                    recv = np.empty(recv_leading_shape, dtype=writer.offsets.dtype)
                    mpi_comm.Recv(recv, source=proc, tag=tag_stream_offsets)
                writer.save_offsets(recv, mpi_comm, dslc, fslc)
                del recv
            if writer.have_gains:
                if proc == 0:
                    recv = writer.gains
                else:
                    recv = np.empty(recv_leading_shape, dtype=writer.gains.dtype)
                    mpi_comm.Recv(recv, source=proc, tag=tag_stream_gains)
                writer.save_gains(recv, mpi_comm, dslc, fslc)
                del recv

            # Compressed bytes
            if proc == 0:
                recv = writer.compressed
            else:
                recv = np.empty(global_process_nbytes[proc], dtype=np.uint8)
                mpi_comm.Recv(recv, source=proc, tag=tag_comp)
            dslc = (slice(0, global_process_nbytes[proc]),)
            fslc = (
                slice(
                    comp_doff[proc],
                    comp_doff[proc] + global_process_nbytes[proc],
                ),
            )
            writer.save_compressed(recv, mpi_comm, dslc, fslc)
            del recv
        elif proc == rank:
            # We are sending.
            mpi_comm.Send(writer.starts.astype(np.int64), dest=0, tag=tag_starts)
            mpi_comm.Send(writer.nbytes, dest=0, tag=tag_nbytes)
            if writer.offsets is not None:
                mpi_comm.Send(writer.offsets, dest=0, tag=tag_stream_offsets)
            if writer.gains is not None:
                mpi_comm.Send(writer.gains, dest=0, tag=tag_stream_gains)
            mpi_comm.Send(writer.compressed, dest=0, tag=tag_comp)
