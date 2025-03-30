# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Common I/O functions.

This module contains helper tools used by multiple I/O formats.

"""

import numpy as np

from .mpi import MPI
from .utils import keep_select, function_timer, select_keep_indices, log


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


def extract_proc_buffers(reader, comm, dist, proc, global_leading_shape, keep):
    """Helper function to extract the buffers for a single process."""
    # The range of the leading dimension on this process.
    send_range = dist[proc]
    send_leading_shape = (int(send_range[1] - send_range[0]),) + global_leading_shape[
        1:
    ]

    # The helper datasets all have the same slab definitions
    dslc = tuple([slice(0, x) for x in send_leading_shape])
    fslc = (slice(send_range[0], send_range[0] + send_leading_shape[0]),) + tuple(
        [slice(0, x) for x in send_leading_shape[1:]]
    )

    # If we are using the "keep" array to select streams, slice that
    # to cover only data for this process.
    if keep is None:
        proc_keep = None
    else:
        proc_keep = keep[dslc]

    # Stream starts
    raw_starts = reader.load_starts(comm, fslc, dslc)

    # Stream nbytes
    raw_nbytes = reader.load_nbytes(comm, fslc, dslc)

    # Offsets and gains for type conversions
    raw_offsets = reader.load_offsets(comm, fslc, dslc)
    raw_gains = reader.load_gains(comm, fslc, dslc)

    # Compressed bytes.  Apply our stream selection and load just those
    # streams we are keeping for this process.
    dcomp = reader.compressed_dataset
    proc_compressed, proc_starts, proc_keep_indices = read_compressed_dataset_slice(
        dcomp, proc_keep, raw_starts, raw_nbytes
    )

    if proc_starts is None:
        # This rank has no data after masking
        proc_shape = None
    else:
        # The data shape after masking is always reduced to a 2D array of
        # streams, with the returned indices describing the location of each
        # stream in the original array.
        proc_shape = tuple(proc_starts.shape)

    # Cut our other arrays to only include the indices selected by the keep
    # mask.
    proc_nbytes = select_keep_indices(raw_nbytes, proc_keep_indices)
    proc_offsets = select_keep_indices(raw_offsets, proc_keep_indices)
    proc_gains = select_keep_indices(raw_gains, proc_keep_indices)

    return (
        proc_shape,
        proc_keep,
        proc_keep_indices,
        proc_compressed,
        proc_starts,
        proc_nbytes,
        proc_offsets,
        proc_gains,
    )


def send_proc_buffers(
    comm,
    proc,
    proc_shape,
    proc_keep_indices,
    proc_compressed,
    proc_starts,
    proc_nbytes,
    proc_offsets,
    proc_gains,
    is_64bit=False,
):
    """Helper function to send the buffers for one process's data."""
    # Send to correct process.
    buffers = [
        (proc_starts, MPI.INT64_T),
        (proc_nbytes, MPI.INT64_T),
        (proc_compressed, MPI.BYTE),
    ]
    if proc_offsets is not None:
        if is_64bit:
            buffers.append((proc_offsets, MPI.DOUBLE))
        else:
            buffers.append((proc_offsets, MPI.FLOAT))
    if proc_gains is not None:
        if is_64bit:
            buffers.append((proc_gains, MPI.DOUBLE))
        else:
            buffers.append((proc_gains, MPI.FLOAT))

    # Send two pieces of information needed to receive further data: the
    # local shape and the keep indices.  Since proc_shape is small and
    # keep indices are a bool array, we send these with lower-case `isend`,
    # which pickles under the hood.
    max_n_send = 7
    tag_base = max_n_send * proc

    for imsg, obj in enumerate([proc_shape, proc_keep_indices]):
        msg_tag = tag_base + imsg
        comm.send(obj, dest=proc, tag=msg_tag)

    if proc_shape is not None:
        # This process has some data
        for itag, (buf, buftype) in enumerate(buffers):
            msg_tag = tag_base + 2 + itag
            comm.Send([buf, buftype], dest=proc, tag=msg_tag)


def receive_proc_buffers(
    comm,
    proc,
    stream_size,
    is_64bit=False,
    offsetgain=False,
):
    """Helper function to receive the buffers for a single process."""
    # First receive the shape and keep indices
    max_n_recv = 7
    tag_base = max_n_recv * proc

    msg_tag = tag_base
    proc_shape = comm.recv(source=0, tag=msg_tag)

    msg_tag += 1
    proc_keep_indices = comm.recv(source=0, tag=msg_tag)

    local_shape = None
    keep_indices = None
    local_starts = None
    stream_nbytes = None
    compressed = None
    stream_offsets = None
    stream_gains = None
    if proc_shape is not None:
        # This process has some data
        local_shape = proc_shape + (stream_size,)
        keep_indices = proc_keep_indices

        msg_tag += 1
        local_starts = np.empty(proc_shape, dtype=np.int64)
        comm.Recv([local_starts, MPI.INT64_T], source=0, tag=msg_tag)

        msg_tag += 1
        stream_nbytes = np.empty(proc_shape, dtype=np.int64)
        comm.Recv([stream_nbytes, MPI.INT64_T], source=0, tag=msg_tag)

        total_bytes = np.sum(stream_nbytes)
        compressed = np.empty(total_bytes, dtype=np.uint8)
        msg_tag += 1
        comm.Recv([compressed, MPI.BYTE], source=0, tag=msg_tag)

        if offsetgain:
            # We have floating point data with offsets / gains.
            msg_tag += 1
            if is_64bit:
                stream_offsets = np.empty(proc_shape, dtype=np.float64)
                comm.Recv([stream_offsets, MPI.DOUBLE], source=0, tag=msg_tag)
            else:
                stream_offsets = np.empty(proc_shape, dtype=np.float32)
                comm.Recv([stream_offsets, MPI.FLOAT], source=0, tag=msg_tag)

            msg_tag += 1
            if is_64bit:
                stream_gains = np.empty(proc_shape, dtype=np.float64)
                comm.Recv([stream_gains, MPI.DOUBLE], source=0, tag=msg_tag)
            else:
                stream_gains = np.empty(proc_shape, dtype=np.float32)
                comm.Recv([stream_gains, MPI.FLOAT], source=0, tag=msg_tag)
    return (
        local_shape,
        keep_indices,
        local_starts,
        stream_nbytes,
        compressed,
        stream_offsets,
        stream_gains,
    )


@function_timer
def read_send_compressed(
    reader, global_shape, n_channel, keep=None, mpi_comm=None, mpi_dist=None
):
    """Read data on one process and distribute.

    Args:
        reader (class):  The Reader class instance.
        global_shape (tuple):  Global shape of the uncompressed array.
        n_channel (int):  The number of compressed 32bit channels.
        keep (array):  Boolean array of streams to keep.
        mpi_comm (MPI.Comm):  The MPI communicator or None.
        mpi_dist (dict):  The distribution of the leading dimension over processes.

    Returns:
        (tuple):  The data and metadata

    """
    if mpi_comm is None:
        nproc = 1
        rank = 0
        comm = None
    else:
        # If the blocks of compressed data exceed 2^30 elements in total per process,
        # they might hit MPI limitations on the communication message sizes.  Work
        # around that here.
        try:
            from mpi4py.util import pkl5

            comm = pkl5.Intracomm(mpi_comm)
        except Exception:
            comm = mpi_comm
        nproc = comm.size
        rank = comm.rank

    global_leading_shape = global_shape[:-1]
    stream_size = global_shape[-1]

    local_shape = None
    local_starts = None
    stream_nbytes = None
    compressed = None
    stream_offsets = None
    stream_gains = None
    keep_indices = None

    is_64bit = False
    if n_channel == 2:
        is_64bit = True

    # Check that the reader dtypes are consistent for the offsets and gains,
    # if those are used.
    offsets_and_gains = False
    if reader.stream_off_dtype is not None:
        offsets_and_gains = True
        if reader.stream_gain_dtype is None:
            raise RuntimeError(
                "Reader stream offsets / gains must both be None or valid"
            )
        if reader.stream_off_dtype != reader.stream_gain_dtype:
            raise RuntimeError(
                "Reader stream offsets and gains should be the same dtype"
            )
        if reader.stream_off_dtype == np.dtype(np.float64):
            # Offsets and gains are 64bit- does that match the number of channels?
            if not is_64bit:
                raise RuntimeError(
                    "Reader offsets / gains are float64, but n_channel != 2"
                )

    # One process reads and sends.
    # The rank zero process will read data and send to the other
    # processes.  Keep a handle to the asynchronous send buffers
    # and delete them after the sends are complete.
    for proc in range(nproc):
        if rank == 0:
            (
                proc_shape,
                proc_keep,
                proc_keep_indices,
                proc_compressed,
                proc_starts,
                proc_nbytes,
                proc_offsets,
                proc_gains,
            ) = extract_proc_buffers(
                reader, comm, mpi_dist, proc, global_leading_shape, keep
            )

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
                send_proc_buffers(
                    comm,
                    proc,
                    proc_shape,
                    proc_keep_indices,
                    proc_compressed,
                    proc_starts,
                    proc_nbytes,
                    proc_offsets,
                    proc_gains,
                )
        elif proc == rank:
            (
                local_shape,
                keep_indices,
                local_starts,
                stream_nbytes,
                compressed,
                stream_offsets,
                stream_gains,
            ) = receive_proc_buffers(
                comm,
                proc,
                stream_size,
                is_64bit=is_64bit,
                offsetgain=offsets_and_gains,
            )

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
    writer,
    global_leading_shape,
    global_process_nbytes,
    n_channel,
    mpi_comm=None,
    mpi_dist=None,
):
    """Receive data on one process and write.

    Args:
        writer (class):  The Writer class instance.
        global_leading_shape (tuple):  Global shape of leading array dimensions.
        global_process_nbytes (list):  Number of bytes on each process.
        mpi_comm (MPI.Comm):  The MPI communicator or None.
        mpi_dist (dict):  The distribution of the leading dimension over processes.

    Returns:
        (tuple):  The data and metadata

    """
    if mpi_comm is None:
        nproc = 1
        rank = 0
        comm = None
    else:
        # If the blocks of compressed data exceed 2^30 elements in total per process,
        # they might hit MPI limitations on the communication message sizes.  Work
        # around that here.
        try:
            from mpi4py.util import pkl5

            comm = pkl5.Intracomm(mpi_comm)
        except Exception:
            comm = mpi_comm
        nproc = comm.size
        rank = comm.rank

    # Compute the byte offset of each process's data
    comp_doff = list()
    coff = 0
    for proc in range(nproc):
        comp_doff.append(coff)
        coff += global_process_nbytes[proc]

    is_64bit = n_channel == 2

    # Check that the writer dtypes are consistent for the offsets and gains,
    # if those are used.
    offsets_and_gains = False
    if writer.offsets is not None:
        offsets_and_gains = True
        if writer.gains is None:
            raise RuntimeError(
                "Writer stream offsets / gains must both be None or valid"
            )
        if writer.offsets.dtype != writer.gains.dtype:
            raise RuntimeError(
                "Writer stream offsets and gains should be the same dtype"
            )
        if writer.offsets.dtype == np.dtype(np.float64):
            # Offsets and gains are 64bit- does that match the number of channels?
            if not is_64bit:
                raise RuntimeError(
                    "Writer offsets / gains are float64, but n_channel != 2"
                )

    for proc in range(nproc):
        # Set up communication tags for the buffers we will send / receive.  The
        # buffers are sent from unique processes, so we can re-use the same tags
        # for each sending process.
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

            # The number of streams we expect from this process
            n_recv = np.prod(recv_leading_shape)
            if n_recv == 0:
                # No data, nothing to receive
                continue

            # The next 4 datasets all have the same slab definitions
            dslc = tuple([slice(0, x) for x in recv_leading_shape])
            fslc = (
                slice(recv_range[0], recv_range[0] + recv_leading_shape[0]),
            ) + tuple([slice(0, x) for x in recv_leading_shape[1:]])

            # Stream starts
            if proc == 0:
                # Copy our local data.
                recv = writer.starts.astype(np.int64)
            else:
                # Receive
                recv = np.empty(recv_leading_shape, dtype=np.int64)
                comm.Recv([recv, MPI.INT64_T], source=proc, tag=tag_starts)
            writer.save_starts(recv, comm, dslc, fslc)
            del recv

            # Stream nbytes
            if proc == 0:
                recv = writer.nbytes.astype(np.int64)
            else:
                recv = np.empty(recv_leading_shape, dtype=np.int64)
                comm.Recv([recv, MPI.INT64_T], source=proc, tag=tag_nbytes)
            writer.save_nbytes(recv, comm, dslc, fslc)
            del recv

            # Offsets and gains for type conversions
            if offsets_and_gains:
                if proc == 0:
                    if is_64bit:
                        recv = writer.offsets.astype(np.float64)
                    else:
                        recv = writer.offsets.astype(np.float32)
                else:
                    if is_64bit:
                        recv = np.empty(recv_leading_shape, dtype=np.float64)
                        comm.Recv(
                            [recv, MPI.DOUBLE], source=proc, tag=tag_stream_offsets
                        )
                    else:
                        recv = np.empty(recv_leading_shape, dtype=np.float32)
                        comm.Recv(
                            [recv, MPI.FLOAT], source=proc, tag=tag_stream_offsets
                        )
                writer.save_offsets(recv, comm, dslc, fslc)
                del recv

                if proc == 0:
                    if is_64bit:
                        recv = writer.gains.astype(np.float64)
                    else:
                        recv = writer.gains.astype(np.float32)
                else:
                    if is_64bit:
                        recv = np.empty(recv_leading_shape, dtype=np.float64)
                        comm.Recv([recv, MPI.DOUBLE], source=proc, tag=tag_stream_gains)
                    else:
                        recv = np.empty(recv_leading_shape, dtype=np.float32)
                        comm.Recv([recv, MPI.FLOAT], source=proc, tag=tag_stream_gains)
                writer.save_gains(recv, comm, dslc, fslc)
                del recv

            # Compressed bytes
            if proc == 0:
                recv = writer.compressed
            else:
                recv = np.empty(global_process_nbytes[proc], dtype=np.uint8)
                comm.Recv([recv, MPI.BYTE], source=proc, tag=tag_comp)
            dslc = (slice(0, global_process_nbytes[proc]),)
            fslc = (
                slice(
                    comp_doff[proc],
                    comp_doff[proc] + global_process_nbytes[proc],
                ),
            )
            writer.save_compressed(recv, comm, dslc, fslc)
            del recv
        elif proc == rank:
            # We are sending.
            send_range = mpi_dist[proc]
            if send_range[1] - send_range[0] == 0:
                # We have no data
                continue
            comm.Send(writer.starts.astype(np.int64), dest=0, tag=tag_starts)
            comm.Send(writer.nbytes.astype(np.int64), dest=0, tag=tag_nbytes)
            if writer.offsets is not None:
                if is_64bit:
                    comm.Send(
                        [writer.offsets, MPI.DOUBLE], dest=0, tag=tag_stream_offsets
                    )
                else:
                    comm.Send(
                        [writer.offsets, MPI.FLOAT], dest=0, tag=tag_stream_offsets
                    )
            if writer.gains is not None:
                if is_64bit:
                    comm.Send([writer.gains, MPI.DOUBLE], dest=0, tag=tag_stream_gains)
                else:
                    comm.Send([writer.gains, MPI.FLOAT], dest=0, tag=tag_stream_gains)
            comm.Send(writer.compressed, dest=0, tag=tag_comp)
