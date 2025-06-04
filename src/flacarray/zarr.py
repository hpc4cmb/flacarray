# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for writing/reading FlacArray data to/from Zarr files

The schema within a Zarr Group is versioned with a simple integer.
The `write_zarr` function always writes the latest version of the
format, but the `read_zarr` function can read the current and past
versions.

"""
import importlib

import numpy as np

try:
    import zarr

    have_zarr = True
except ImportError:
    have_zarr = False

from . import __version__ as flacarray_version
from .compress import array_compress
from .io_common import receive_write_compressed
from .mpi import global_array_properties, global_bytes
from .utils import function_timer


class ZarrGroup(object):
    """Wrapper class containing an open Zarr Group.

    The named object is a file opened in the specified mode on the root process.
    On other processes the handle will be None.

    Args:
        name (str):  The filesystem path.
        mode (str):  The opening mode.
        comm (MPI.Comm):  The MPI communicator or None.

    """
    def __init__(self, name, mode, comm=None):
        self.handle = None
        if comm is None or comm.rank == 0:
            self.handle = zarr.open_group(name, mode=mode)
        if comm is not None:
            comm.barrier()

    def close(self):
        if hasattr(self, "handle") and self.handle is not None:
            self.handle.store.close()
            del self.handle
            self.handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self.handle

    def __exit__(self, *args):
        self.close()


class WriterZarr:
    """Helper class for the common writer function."""

    def __init__(
        self,
        global_stream_starts,
        stream_nbytes,
        stream_comp,
        stream_offsets,
        stream_gains,
        dataset_starts,
        dataset_nbytes,
        dataset_comp,
        dataset_offsets,
        dataset_gains,
    ):
        self._starts = global_stream_starts
        self._nbytes = stream_nbytes
        self._comp = stream_comp
        self._offsets = stream_offsets
        self._gains = stream_gains
        self._dstarts = dataset_starts
        self._dnbytes = dataset_nbytes
        self._dcomp = dataset_comp
        self._doffsets = dataset_offsets
        self._dgains = dataset_gains

    @property
    def starts(self):
        return self._starts

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def compressed(self):
        return self._comp

    @property
    def offsets(self):
        return self._offsets

    @property
    def gains(self):
        return self._gains

    @property
    def have_offsets(self):
        return self._doffsets is not None

    @property
    def have_gains(self):
        return self._gains is not None

    def save(self, dset, buf, mpi_comm, dslc, fslc):
        rank = 0
        if mpi_comm is not None:
            rank = mpi_comm.rank
        if dset is None or rank != 0:
            return
        dset[fslc] = buf[dslc]

    def save_starts(self, buf, mpi_comm, dslc, fslc):
        return self.save(self._dstarts, buf, mpi_comm, dslc, fslc)

    def save_nbytes(self, buf, mpi_comm, dslc, fslc):
        return self.save(self._dnbytes, buf, mpi_comm, dslc, fslc)

    def save_offsets(self, buf, mpi_comm, dslc, fslc):
        return self.save(self._doffsets, buf, mpi_comm, dslc, fslc)

    def save_gains(self, buf, mpi_comm, dslc, fslc):
        return self.save(self._dgains, buf, mpi_comm, dslc, fslc)

    def save_compressed(self, buf, mpi_comm, dslc, fslc):
        return self.save(self._dcomp, buf, mpi_comm, dslc, fslc)


@function_timer
def write_compressed(
    zgrp,
    leading_shape,
    global_leading_shape,
    stream_size,
    stream_starts,
    global_stream_starts,
    stream_nbytes,
    stream_offsets,
    stream_gains,
    compressed,
    n_channels,
    local_nbytes,
    global_nbytes,
    global_process_nbytes,
    mpi_comm,
    mpi_dist,
):
    """Write compressed data to a Zarr group.

    The various input parameters are computed either during construction of a
    FlacArray, or during the `write_array` convenience function.

    We make the assumption that the Group may already have attributes associated
    with the calling code, and we do not want to modify or introduce conflicts with
    those.  We will make several datasets inside this group, and we will place most
    of our attributes on the dataset associated with the starting byte offsets.  We
    intentionally write the "flacarray_format_version" attribute to the top level
    group so that we can parse that and call the correct version of the read function.

    Args:
        zgrp (zarr.Group):  The Group to use.
        leading_shape (tuple):  Shape of the local leading dimensions.
        global_leading_shape (tuple):  Global shape of the local leading dimensions.
        stream_size (int):  The length of each stream.
        stream_starts (array):  The local starting byte offsets for each stream.
        global_stream_starts (array):  The global starting byte offsets for each stream.
        stream_nbytes (array):  The local number of bytes for each stream.
        stream_offsets (array):  The offsets used in int conversion.
        stream_gains (array):  The gains used in int conversion.
        compressed (array):  The compressed bytes.
        n_channels (int):  The number of FLAC channels used (1 or 2).
        local_nbytes (int):  The total number of local compressed bytes.
        global_nbytes (int):  The total global compressed bytes.
        global_process_nbytes (list):  The number of compressed bytes on each process.
        mpi_comm (MPI.Comm):  The MPI communicator.
        mpi_dist (list):  The range of the leading dimension on each process.

    Returns:
        None

    """
    if not have_zarr:
        raise RuntimeError("zarr is not importable, cannot write to a zarr.Group")

    # Writer is currently using version 1
    from .zarr_load_v1 import zarr_names as znames

    comm = mpi_comm
    if comm is None:
        rank = 0
    else:
        rank = comm.rank

    dstarts = None
    dbytes = None
    dcomp = None
    dsoff = None
    dsgain = None

    if rank == 0:
        # This process is participating.  Write the format version string
        # to the top-level group.
        zgrp.attrs["flacarray_format_version"] = "1"
        zgrp.attrs["flacarray_software_version"] = flacarray_version
        zgrp.attrs[znames["flac_channels"]] = f"{n_channels}"

        # Create the datasets.  We create the start bytes and auxiliary datasets first
        # and attach any metadata keys to the start bytes dataset (which is always
        # guaranteed to exist).  We also create a dataset storing the number of bytes
        # in each stream.  Although this can technically be computed using the total
        # number of compressed bytes and the stream starting bytes, it greatly
        # improves the convenience of loading data back in.

        # Zarr 3.0 requires shapes to be tuples of int
        if len(global_leading_shape) == 0:
            z_global_leading_shape = (1,)
        else:
            z_global_leading_shape = tuple([int(x) for x in global_leading_shape])
        z_global_nbytes = (int(global_nbytes),)

        if hasattr(zgrp, "create_array"):
            # Zarr-3
            create_func = zgrp.create_array
        else:
            # Zarr-2
            create_func = zgrp.create_dataset

        # The starting bytes of each stream
        dstarts = create_func(
            znames["stream_starts"],
            shape=z_global_leading_shape,
            dtype=np.int64,
        )
        dstarts.attrs[znames["stream_size"]] = stream_size

        # The number of bytes in each stream
        dbytes = create_func(
            znames["stream_bytes"],
            shape=z_global_leading_shape,
            dtype=np.int64,
        )

        # The stream offsets and gains are optional, depending on the original
        # array type.
        if stream_offsets is not None:
            dsoff = create_func(
                znames["stream_offsets"],
                shape=z_global_leading_shape,
                dtype=stream_offsets.dtype,
            )
        else:
            dsoff = None
        if stream_gains is not None:
            dsgain = create_func(
                znames["stream_gains"],
                shape=z_global_leading_shape,
                dtype=stream_gains.dtype,
            )
        else:
            dsgain = None

        # Always have compressed bytes
        dcomp = create_func(
            znames["compressed"],
            shape=z_global_nbytes,
            dtype=np.uint8,
        )

    # Use the common writing function
    writer = WriterZarr(
        global_stream_starts,
        stream_nbytes,
        compressed,
        stream_offsets,
        stream_gains,
        dstarts,
        dbytes,
        dcomp,
        dsoff,
        dsgain,
    )
    receive_write_compressed(
        writer,
        global_leading_shape,
        global_process_nbytes,
        n_channels,
        mpi_comm=mpi_comm,
        mpi_dist=mpi_dist,
    )


@function_timer
def write_array(
    arr, zgrp, level=5, quanta=None, precision=None, mpi_comm=None, use_threads=False
):
    """Compress a numpy array and write to an Zarr group.

    This function is useful if you do not need to access the compressed array in memory
    and only wish to write it directly to Zarr files.  The input array is compressed
    and then the `write_compressed()` function is called.

    If the input array is int32 or int64, the compression is lossless and the compressed
    bytes and ancillary data is written to datasets within the output group.  If the
    array is float32 or float64, either the `quanta` or `precision` must be specified.
    See discussion in the `FlacArray` class documentation about how the offsets and
    gains are computed for a given quanta.  The offsets and gains are also written as
    datasets within the output group.

    Args:
        arr (array):  The input numpy array.
        zgrp (zarr.Group):  The Group to use.
        level (int):  Compression level (0-8).
        quanta (float, array):  For floating point data, the floating point
            increment of each 32bit integer value.  Optionally an iterable of
            increments, one per stream.
        precision (int, array):  Number of significant digits to retain in
            float-to-int conversion.  Alternative to `quanta`.  Optionally an
            iterable of values, one per stream.
        mpi_comm (MPI.Comm):  If specified, the input array is assumed to be
            distributed across the communicator at the leading dimension.  The
            local piece of the array is passed in on each process.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

    Returns:
        None

    """
    if not have_zarr:
        raise RuntimeError("zarr is not importable, cannot write to zarr.Group")

    # Get the global shape of the array
    global_props = global_array_properties(arr.shape, mpi_comm=mpi_comm)
    global_shape = global_props["shape"]
    mpi_dist = global_props["dist"]

    # Get the number of channels
    if arr.dtype == np.dtype(np.int64) or arr.dtype == np.dtype(np.float64):
        n_channels = 2
    else:
        n_channels = 1

    # Compress our local piece of the array
    compressed, starts, nbytes, offsets, gains = array_compress(
        arr, level=level, quanta=quanta, precision=precision, use_threads=use_threads
    )

    local_nbytes = compressed.nbytes
    global_nbytes, global_proc_bytes, global_starts = global_bytes(
        local_nbytes, starts, mpi_comm
    )
    stream_size = arr.shape[-1]

    if len(arr.shape) == 1:
        leading_shape = (1,)
    else:
        leading_shape = arr.shape[:-1]
    global_leading_shape = global_shape[:-1]

    write_compressed(
        zgrp,
        leading_shape,
        global_leading_shape,
        stream_size,
        starts,
        global_starts,
        nbytes,
        offsets,
        gains,
        compressed,
        n_channels,
        local_nbytes,
        global_nbytes,
        global_proc_bytes,
        mpi_comm,
        mpi_dist,
    )


@function_timer
def read_compressed(zgrp, keep=None, mpi_comm=None, mpi_dist=None):
    """Load compressed data from a Zarr Group.

    This function acts as a dispatch to the correct version of the reading
    function.  The function is selected based on the format version string
    in the data.

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
    if not have_zarr:
        raise RuntimeError("zarr is not importable, cannot write to a Zarr Group")

    format_version = None
    if zgrp is not None:
        if "flacarray_format_version" in zgrp.attrs:
            format_version = zgrp.attrs["flacarray_format_version"]
    if mpi_comm is not None:
        format_version = mpi_comm.bcast(format_version, root=0)
    if format_version is None:
        raise RuntimeError("Zarr Group does not contain a FlacArray")

    mod_name = f".zarr_load_v{format_version}"
    mod = importlib.import_module(mod_name, package="flacarray")
    read_func = getattr(mod, "read_compressed")
    return read_func(
        zgrp,
        keep=keep,
        mpi_comm=mpi_comm,
        mpi_dist=mpi_dist,
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
    """Load a numpy array from a compressed Zarr group.

    This function is useful if you do not need to store a compressed representation
    of the array in memory.  Each stream will be read individually from the file and
    the desired slice decompressed.  This avoids storing the full compressed data.

    This function acts as a dispatch to the correct version of the reading function.
    The function is selected based on the format version string in the data.

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
        (array):  The loaded and decompressed data OR the array and the kept indices.

    """
    if not have_zarr:
        raise RuntimeError("zarr is not importable, cannot write to a Zarr Group")

    format_version = None
    if zgrp is not None:
        if "flacarray_format_version" in zgrp.attrs:
            format_version = zgrp.attrs["flacarray_format_version"]
    if mpi_comm is not None:
        format_version = mpi_comm.bcast(format_version, root=0)
    if format_version is None:
        raise RuntimeError("Zarr Group does not contain a FlacArray")

    mod_name = f".zarr_load_v{format_version}"
    mod = importlib.import_module(mod_name, package="flacarray")
    read_func = getattr(mod, "read_array")
    return read_func(
        zgrp,
        keep=keep,
        stream_slice=stream_slice,
        keep_indices=keep_indices,
        mpi_comm=mpi_comm,
        mpi_dist=mpi_dist,
        use_threads=use_threads,
        no_flatten=False,
    )
