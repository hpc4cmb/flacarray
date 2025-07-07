# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Loading functions for HDF5 format version 1.

This module should only be imported on-demand by the higher-level read / write
functions.

"""
import numpy as np

from .decompress import array_decompress
from .hdf5_utils import hdf5_use_serial
from .mpi import distribute_and_verify
from .io_common import (
    read_send_compressed,
    select_keep_indices,
    read_compressed_dataset_slice,
)
from .utils import function_timer


"""The dataset and attribute names."""
hdf5_names = {
    "compressed": "compressed",
    "stream_starts": "stream_starts",
    "stream_bytes": "stream_bytes",
    "stream_size": "stream_size",
    "stream_offsets": "stream_offsets",
    "stream_gains": "stream_gains",
    "flac_channels": "flac_channels",
}


class ReaderHDF5:
    """Helper class for the common reader function."""

    def __init__(
        self,
        dataset_starts,
        dataset_nbytes,
        dataset_comp,
        dataset_offsets,
        dataset_gains,
        offsets_dtype,
        gains_dtype,
    ):
        self._starts = dataset_starts
        self._nbytes = dataset_nbytes
        self._comp = dataset_comp
        self._offsets = dataset_offsets
        self._gains = dataset_gains
        self._offsets_dtype = offsets_dtype
        self._gains_dtype = gains_dtype

    @property
    def compressed_dataset(self):
        return self._comp

    @property
    def stream_off_dtype(self):
        return self._offsets_dtype

    @property
    def stream_gain_dtype(self):
        return self._gains_dtype

    def load(self, dset, mpi_comm, fslc, dslc):
        rank = 0
        if mpi_comm is not None:
            rank = mpi_comm.rank
        if dset is None or rank != 0:
            return None
        shape = tuple([x.stop - x.start for x in dslc])
        raw = np.empty(shape, dtype=dset.dtype)
        dset.read_direct(raw, fslc, dslc)
        return raw

    def load_starts(self, mpi_comm, fslc, dslc):
        return self.load(self._starts, mpi_comm, fslc, dslc)

    def load_nbytes(self, mpi_comm, fslc, dslc):
        return self.load(self._nbytes, mpi_comm, fslc, dslc)

    def load_offsets(self, mpi_comm, fslc, dslc):
        return self.load(self._offsets, mpi_comm, fslc, dslc)

    def load_gains(self, mpi_comm, fslc, dslc):
        return self.load(self._gains, mpi_comm, fslc, dslc)


@function_timer
def read_compressed(hgrp, keep=None, mpi_comm=None, mpi_dist=None):
    """Load compressed data from an HDF group.

    If `stream_slice` is specified, the returned array will have only that
    range of samples in the final dimension.

    If `keep` is specified, this should be a boolean array with the same shape
    as the leading dimensions of the original array.  True values in this array
    indicate that the stream should be kept.

    If `keep` is specified, the returned array WILL NOT have the same shape as
    the original.  Instead it will be a 2D array of decompressed streams- the
    streams corresponding to True values in the `keep` mask.

    Args:
        hgrp (h5py.Group):  The group to read.
        keep (array):  Bool array of streams to keep in the decompression.
        mpi_comm (MPI.Comm):  The optional MPI communicator over which to distribute
            the leading dimension of the array.
        mpi_dist (list):  The optional list of tuples specifying the first / last
            element of the leading dimension to assign to each process.

    Returns:
        (tuple):  The compressed data and metadata.

    """
    use_serial = hdf5_use_serial(hgrp, mpi_comm)

    if mpi_comm is None:
        nproc = 1
        rank = 0
    else:
        nproc = mpi_comm.size
        rank = mpi_comm.rank

    # Metadata variables that we may need to communicate later
    stream_size = None
    global_shape = None
    global_nbytes = None
    stream_off_dtype = None
    stream_gain_dtype = None
    n_channel = None

    # Dataset handles (only valding on reading processes)
    dstarts = None
    dbytes = None
    dsoff = None
    dsgain = None
    dcomp = None

    if rank == 0 or not use_serial:
        # This process is participating.
        # Double check that we can load this format.
        ver = int(hgrp.attrs["flacarray_format_version"])
        if ver != 1:
            msg = f"Version 1 loader called with version {ver} data"
            raise RuntimeError(msg)

        # Get a handle to all the datasets, and extract some metadata.
        n_channel = int(hgrp.attrs[hdf5_names["flac_channels"]])
        dstarts = hgrp[hdf5_names["stream_starts"]]
        stream_size = int(dstarts.attrs[hdf5_names["stream_size"]])
        global_shape = dstarts.shape + (stream_size,)
        dbytes = hgrp[hdf5_names["stream_bytes"]]
        dsoff = None
        if hdf5_names["stream_offsets"] in hgrp:
            dsoff = hgrp[hdf5_names["stream_offsets"]]
            stream_off_dtype = np.dtype(dsoff.dtype)
        dsgain = None
        if hdf5_names["stream_gains"] in hgrp:
            dsgain = hgrp[hdf5_names["stream_gains"]]
            stream_gain_dtype = np.dtype(dsgain.dtype)
        dcomp = hgrp[hdf5_names["compressed"]]
        global_nbytes = dcomp.size

    if nproc > 1 and use_serial:
        # Not every process is reading- communicate some of the metadata loaded
        # above.
        stream_size = mpi_comm.bcast(stream_size, root=0)
        global_shape = mpi_comm.bcast(global_shape, root=0)
        global_nbytes = mpi_comm.bcast(global_nbytes, root=0)
        stream_gain_dtype = mpi_comm.bcast(stream_gain_dtype, root=0)
        stream_off_dtype = mpi_comm.bcast(stream_off_dtype, root=0)
        n_channel = mpi_comm.bcast(n_channel, root=0)
    global_leading_shape = global_shape[:-1]

    # Compute or verify the MPI distribution for the global leading dimension
    mpi_dist = distribute_and_verify(mpi_comm, global_shape[0], mpi_dist=mpi_dist)

    # Local data buffers we will load from the file.
    local_shape = None
    local_starts = None
    stream_nbytes = None
    compressed = None
    stream_offsets = None
    stream_gains = None
    keep_indices = None

    if use_serial:
        # Use the common function for reading data and communicating it.
        reader = ReaderHDF5(
            dstarts, dbytes, dcomp, dsoff, dsgain, stream_off_dtype, stream_gain_dtype
        )
        (
            local_shape,
            local_starts,
            stream_nbytes,
            compressed,
            stream_offsets,
            stream_gains,
            keep_indices,
        ) = read_send_compressed(
            reader,
            global_shape,
            n_channel,
            keep=keep,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )
    else:
        # We are using parallel HDF5.  All processes have a handle to the dataset
        # from above, and each process reads its local slice.
        ds_range = mpi_dist[rank]
        leading_shape = (ds_range[1] - ds_range[0],) + global_leading_shape[1:]
        local_shape = leading_shape + (stream_size,)

        # The helper datasets all have the same slab definitions
        dslc = tuple([slice(0, x) for x in leading_shape])
        hslc = (slice(ds_range[0], ds_range[0] + leading_shape[0]),) + tuple(
            [slice(0, x) for x in leading_shape[1:]]
        )

        # If we are using the "keep" array to select streams, slice that
        # to cover only data for this process.
        if keep is None:
            proc_keep = None
        else:
            proc_keep = keep[dslc]

        # Stream starts
        raw_starts = np.empty(leading_shape, dtype=dstarts.dtype)
        dstarts.read_direct(raw_starts, hslc, dslc)

        # Stream nbytes
        raw_nbytes = np.empty(leading_shape, dtype=dstarts.dtype)
        dbytes.read_direct(raw_nbytes, hslc, dslc)

        # Offsets and gains for type conversions
        raw_offsets = None
        if dsoff is not None:
            raw_offsets = np.empty(leading_shape, dtype=stream_off_dtype)
            dsoff.read_direct(raw_offsets, hslc, dslc)
        raw_gains = None
        if dsgain is not None:
            raw_gains = np.empty(leading_shape, dtype=stream_gain_dtype)
            dsgain.read_direct(raw_gains, hslc, dslc)

        # Compressed bytes.  Apply our stream selection and load just those
        # streams we are keeping for this process.
        compressed, local_starts, keep_indices = read_compressed_dataset_slice(
            dcomp, proc_keep, raw_starts, raw_nbytes
        )

        # Cut our other arrays to only include the indices selected by the keep mask.
        stream_nbytes = select_keep_indices(raw_nbytes, keep_indices)
        stream_offsets = select_keep_indices(raw_offsets, keep_indices)
        stream_gains = select_keep_indices(raw_gains, keep_indices)

        if local_starts is None:
            # This rank has no data after masking
            local_shape = None
        else:
            local_shape = local_starts.shape + (stream_size,)

    return (
        local_shape,
        global_shape,
        compressed,
        n_channel,
        local_starts,
        stream_nbytes,
        stream_offsets,
        stream_gains,
        mpi_dist,
        keep_indices,
    )


@function_timer
def read_array(
    hgrp,
    keep=None,
    stream_slice=None,
    keep_indices=False,
    mpi_comm=None,
    mpi_dist=None,
    use_threads=False,
    no_flatten=False,
):
    """Read compressed data directly into an array.

    This function is useful if you do not need to store a compressed representation
    of the array in memory.  Each stream will be read individually from the file and
    the desired slice decompressed.  This avoids storing the full compressed data.

    If `stream_slice` is specified, the returned array will have only that
    range of samples in the final dimension.

    If `keep` is specified, this should be a boolean array with the same shape
    as the leading dimensions of the original array.  True values in this array
    indicate that the stream should be kept.

    If `keep` is specified, the returned array WILL NOT have the same shape as
    the original.  Instead it will be a 2D array of decompressed streams- the
    streams corresponding to True values in the `keep` mask.

    If `keep_indices` is True and `keep` is specified, then an additional list
    is returned containing the indices of each stream that was kept.

    Args:
        hgrp (h5py.Group):  The group to read.
        keep (array):  Bool array of streams to keep in the decompression.
        stream_slice (slice):  A python slice with step size of one, indicating
            the sample range to extract from each stream.
        keep_indices (bool):  If True, also return the original indices of the
            streams.
        mpi_comm (MPI.Comm):  The optional MPI communicator over which to distribute
            the leading dimension of the array.
        mpi_dist (list):  The optional list of tuples specifying the first / last
            element of the leading dimension to assign to each process.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.
        no_flatten (bool):  If True, for single-stream arrays, leave the leading
            dimension of (1,) in the result.


    Returns:
        (array):  The loaded and decompressed data.  Or the array and the kept indices.

    """
    (
        local_shape,
        global_shape,
        compressed,
        n_channel,
        stream_starts,
        stream_nbytes,
        stream_offsets,
        stream_gains,
        mpi_dist,
        indices,
    ) = read_compressed(
        hgrp,
        keep=keep,
        mpi_comm=mpi_comm,
        mpi_dist=mpi_dist,
    )

    first_samp = None
    last_samp = None
    if stream_slice is not None:
        if stream_slice.step is not None and stream_slice.step != 1:
            raise RuntimeError("Only stream slices with a step size of 1 are supported")
        first_samp = stream_slice.start
        last_samp = stream_slice.stop

    arr = array_decompress(
        compressed,
        local_shape[-1],
        stream_starts,
        stream_nbytes,
        stream_offsets=stream_offsets,
        stream_gains=stream_gains,
        first_stream_sample=first_samp,
        last_stream_sample=last_samp,
        is_int64=(n_channel == 2),
        use_threads=use_threads,
        no_flatten=no_flatten,
    )
    if keep_indices:
        return arr, indices
    else:
        return arr
