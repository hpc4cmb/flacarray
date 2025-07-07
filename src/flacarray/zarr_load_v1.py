# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Loading functions for Zarr format version 0.

This module should only be imported on-demand by the higher-level read / write
functions.

"""
import numpy as np

import zarr

from .decompress import array_decompress
from .mpi import distribute_and_verify
from .io_common import read_send_compressed
from .utils import function_timer


"""The dataset and attribute names."""
zarr_names = {
    "compressed": "compressed",
    "stream_starts": "stream_starts",
    "stream_bytes": "stream_bytes",
    "stream_size": "stream_size",
    "stream_offsets": "stream_offsets",
    "stream_gains": "stream_gains",
    "flac_channels": "flac_channels",
}


class ReaderZarr:
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
        raw[dslc] = dset[fslc]
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
def read_compressed(zgrp, keep=None, mpi_comm=None, mpi_dist=None):
    """Load compressed data from an Zarr Group.

    If `keep` is specified, this should be a boolean array with the same shape
    as the leading dimensions of the original array.  True values in this array
    indicate that the stream should be kept.

    If `keep` is specified, the returned array WILL NOT have the same shape as
    the original.  Instead it will be a 2D array of decompressed streams- the
    streams corresponding to True values in the `keep` mask.

    Args:
        zgrp (zarr.Group):  The group to read.
        keep (array):  Bool array of streams to keep in the decompression.
        mpi_comm (MPI.Comm):  The optional MPI communicator over which to distribute
            the leading dimension of the array.
        mpi_dist (list):  The optional list of tuples specifying the first / last
            element of the leading dimension to assign to each process.

    Returns:
        (tuple):  The compressed data and metadata.

    """
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

    if rank == 0:
        # This process is participating.
        # Double check that we can load this format.
        ver = int(zgrp.attrs["flacarray_format_version"])
        if ver != 1:
            msg = f"Version 1 loader called with version {ver} data"
            raise RuntimeError(msg)

        # Get a handle to all the datasets, and extract some metadata.
        n_channel = int(zgrp.attrs[zarr_names["flac_channels"]])
        dstarts = zgrp[zarr_names["stream_starts"]]
        stream_size = int(dstarts.attrs[zarr_names["stream_size"]])
        global_shape = dstarts.shape + (stream_size,)
        dbytes = zgrp[zarr_names["stream_bytes"]]
        dsoff = None
        if zarr_names["stream_offsets"] in zgrp:
            dsoff = zgrp[zarr_names["stream_offsets"]]
            stream_off_dtype = np.dtype(dsoff.dtype)
        dsgain = None
        if zarr_names["stream_gains"] in zgrp:
            dsgain = zgrp[zarr_names["stream_gains"]]
            stream_gain_dtype = np.dtype(dsgain.dtype)
        dcomp = zgrp[zarr_names["compressed"]]
        global_nbytes = dcomp.size

    if nproc > 1:
        # Not every process is reading- communicate some of the metadata loaded
        # above.
        stream_size = mpi_comm.bcast(stream_size, root=0)
        global_shape = mpi_comm.bcast(global_shape, root=0)
        global_nbytes = mpi_comm.bcast(global_nbytes, root=0)
        stream_gain_dtype = mpi_comm.bcast(stream_gain_dtype, root=0)
        stream_off_dtype = mpi_comm.bcast(stream_off_dtype, root=0)
        n_channel = mpi_comm.bcast(n_channel, root=0)

    # Compute or verify the MPI distribution for the global leading dimension
    mpi_dist = distribute_and_verify(mpi_comm, global_shape[0], mpi_dist=mpi_dist)

    # Use the common reader function
    reader = ReaderZarr(
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
    zgrp,
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
        zgrp (zarr.Group):  The group to read.
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
        zgrp,
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
