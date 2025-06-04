# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""Tools for writing/reading FlacArray data to/from HDF5

The schema within an HDF5 Group is versioned with a simple integer.
The `write_hdf5` function always writes the latest version of the
format, but the `read_hdf5` function can read the current and past
versions.

"""
import importlib

import numpy as np

from . import __version__ as flacarray_version
from .compress import array_compress
from .hdf5_utils import have_hdf5, hdf5_use_serial, check_dataset_buffer_size
from .io_common import receive_write_compressed
from .mpi import global_array_properties, global_bytes
from .utils import function_timer, ensure_one_element


class WriterHDF5:
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

    def save(self, dset, buf, mpi_comm, dslc, fslc):
        rank = 0
        if mpi_comm is not None:
            rank = mpi_comm.rank
        if dset is None or rank != 0:
            return
        dset.write_direct(buf, dslc, fslc)

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
    hgrp,
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
    """Write compressed data to an HDF5 group.

    The various input parameters are computed either during construction of a
    FlacArray, or during the `write_array` convenience function.

    We make the assumption that the Group may already have attributes associated
    with the calling code, and we do not want to modify or introduce conflicts with
    those.  We will make several datasets inside this group, and we will place most
    of our attributes on the dataset associated with the starting byte offsets.  We
    intentionally write the "flacarray_format_version" attribute to the top level
    group so that we can parse that and call the correct version of the read function.

    In the case of a single stream, all auxiliary datasets will still contain an
    array (of a single element).

    Args:
        hgrp (h5py.Group):  The Group to use.
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
    if not have_hdf5:
        raise RuntimeError("h5py is not importable, cannot write to HDF5")

    # Writer is currently using version 1
    from .hdf5_load_v1 import hdf5_names as hnames

    comm = mpi_comm

    use_serial = hdf5_use_serial(hgrp, comm)

    if comm is None:
        nproc = 1
        rank = 0
    else:
        nproc = comm.size
        rank = comm.rank

    dstarts = None
    dbytes = None
    dcomp = None
    dsoff = None
    dsgain = None

    aux_global_shape = global_leading_shape
    aux_local_shape = leading_shape
    if len(aux_global_shape) == 0:
        # We have a single stream.  Promote all our auxiliary data
        # to arrays if needed.
        aux_global_shape = (1,)
        aux_local_shape = (1,)
        stream_starts = ensure_one_element(stream_starts, np.int64)
        global_stream_starts = ensure_one_element(global_stream_starts, np.int64)
        stream_nbytes = ensure_one_element(stream_nbytes, np.int64)
        if stream_offsets is not None:
            if n_channels == 2:
                stream_offsets = ensure_one_element(stream_offsets, np.float64)
                stream_gains = ensure_one_element(stream_gains, np.float64)
            else:
                stream_offsets = ensure_one_element(stream_offsets, np.float32)
                stream_gains = ensure_one_element(stream_gains, np.float32)

    if rank == 0 or not use_serial:
        # This process is participating.  Write the format version string
        # to the top-level group.
        hgrp.attrs["flacarray_format_version"] = "1"
        hgrp.attrs["flacarray_software_version"] = flacarray_version
        hgrp.attrs[hnames["flac_channels"]] = f"{n_channels}"

        # Create the datasets.  We create the start bytes and auxiliary datasets first
        # and attach any metadata keys to the start bytes dataset (which is always
        # guaranteed to exist).  We also create a dataset storing the number of bytes
        # in each stream.  Although this can technically be computed using the total
        # number of compressed bytes and the stream starting bytes, it greatly
        # improves the convenience of loading data back in.

        # The starting bytes of each stream
        dstarts = hgrp.create_dataset(
            hnames["stream_starts"],
            aux_global_shape,
            dtype=np.int64,
        )
        dstarts.attrs[hnames["stream_size"]] = stream_size

        # The number of bytes in each stream
        dbytes = hgrp.create_dataset(
            hnames["stream_bytes"],
            aux_global_shape,
            dtype=np.int64,
        )

        # The stream offsets and gains are optional, depending on the original
        # array type.
        if stream_offsets is not None:
            dsoff = hgrp.create_dataset(
                hnames["stream_offsets"],
                aux_global_shape,
                dtype=stream_offsets.dtype,
            )
        else:
            dsoff = None
        if stream_gains is not None:
            dsgain = hgrp.create_dataset(
                hnames["stream_gains"],
                aux_global_shape,
                dtype=stream_gains.dtype,
            )
        else:
            dsgain = None

        # Always have compressed bytes
        dcomp = hgrp.create_dataset(
            hnames["compressed"],
            (global_nbytes,),
            dtype=np.uint8,
        )

    if use_serial:
        # Use the common writing function
        writer = WriterHDF5(
            global_stream_starts.reshape(aux_local_shape),
            stream_nbytes.reshape(aux_local_shape),
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
    else:
        # We are using parallel HDF5.  Every process will write a slice of each
        # dataset.  In this scenario, every process has a non-None handle to the
        # HDF5 Group upon entering this function.  So the datasets have already been
        # created synchronously on all processes above.

        # Compute the byte offset of each process's data
        comp_doff = list()
        coff = 0
        for proc in range(nproc):
            comp_doff.append(coff)
            coff += global_process_nbytes[proc]

        dslc = tuple([slice(0, x) for x in aux_local_shape])
        hslc = (
            slice(
                mpi_dist[rank][0],
                mpi_dist[rank][0] + aux_local_shape[0],
            ),
        ) + tuple([slice(0, x) for x in aux_local_shape[1:]])

        with dstarts.collective:
            dstarts.write_direct(global_stream_starts, dslc, hslc)
        with dbytes.collective:
            dbytes.write_direct(stream_nbytes, dslc, hslc)

        if stream_offsets is not None:
            with dsoff.collective:
                dsoff.write_direct(stream_offsets, dslc, hslc)
        if stream_gains is not None:
            with dsgain.collective:
                dsgain.write_direct(stream_gains, dslc, hslc)

        dslc = (slice(0, global_process_nbytes[rank]),)
        hslc = (slice(comp_doff[rank], comp_doff[rank] + global_process_nbytes[rank]),)
        check_dataset_buffer_size(
            "Parallel write of compressed data", dslc, np.uint8, True
        )
        with dcomp.collective:
            dcomp.write_direct(compressed, dslc, hslc)


@function_timer
def write_array(
    arr, hgrp, level=5, quanta=None, precision=None, mpi_comm=None, use_threads=False
):
    """Compress a numpy array and write to an HDF5 group.

    This function is useful if you do not need to access the compressed array in memory
    and only wish to write it directly to HDF5.  The input array is compressed and then
    the `write_compressed()` function is called.

    If the input array is int32 or int64, the compression is lossless and the compressed
    bytes and ancillary data is written to datasets within the output group.  If the
    array is float32 or float64, either the `quanta` or `precision` must be specified.
    See discussion in the `FlacArray` class documentation about how the offsets and
    gains are computed for a given quanta.  The offsets and gains are also written as
    datasets within the output group.

    Args:
        arr (array):  The input numpy array.
        hgrp (h5py.Group):  The Group to use.
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
    if not have_hdf5:
        raise RuntimeError("h5py is not importable, cannot write to HDF5")

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
        hgrp,
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
def read_compressed(hgrp, keep=None, mpi_comm=None, mpi_dist=None):
    """Load compressed data from HDF5.

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
        hgrp (h5py.Group):  The group to read.
        keep (array):  Bool array of streams to keep in the decompression.
        mpi_comm (MPI.Comm):  The optional MPI communicator over which to distribute
            the leading dimension of the array.
        mpi_dist (list):  The optional list of tuples specifying the first / last
            element of the leading dimension to assign to each process.

    Returns:
        (tuple):  The compressed data and metadata.

    """
    if not have_hdf5:
        raise RuntimeError("h5py is not importable, cannot write to HDF5")

    format_version = None
    if hgrp is not None:
        if "flacarray_format_version" in hgrp.attrs:
            format_version = hgrp.attrs["flacarray_format_version"]
    if mpi_comm is not None:
        format_version = mpi_comm.bcast(format_version, root=0)
    if format_version is None:
        raise RuntimeError("h5py Group does not contain a FlacArray")

    mod_name = f".hdf5_load_v{format_version}"
    mod = importlib.import_module(mod_name, package="flacarray")
    read_func = getattr(mod, "read_compressed")
    return read_func(
        hgrp,
        keep=keep,
        mpi_comm=mpi_comm,
        mpi_dist=mpi_dist,
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
):
    """Load a numpy array from compressed HDF5.

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

    Returns:
        (array):  The loaded and decompressed data OR the array and the kept indices.

    """
    if not have_hdf5:
        raise RuntimeError("h5py is not importable, cannot write to HDF5")

    format_version = None
    if hgrp is not None:
        if "flacarray_format_version" in hgrp.attrs:
            format_version = hgrp.attrs["flacarray_format_version"]
    if mpi_comm is not None:
        format_version = mpi_comm.bcast(format_version, root=0)
    if format_version is None:
        raise RuntimeError("h5py Group does not contain a FlacArray")

    mod_name = f".hdf5_load_v{format_version}"
    mod = importlib.import_module(mod_name, package="flacarray")
    read_func = getattr(mod, "read_array")
    return read_func(
        hgrp,
        keep=keep,
        stream_slice=stream_slice,
        keep_indices=keep_indices,
        mpi_comm=mpi_comm,
        mpi_dist=mpi_dist,
        use_threads=use_threads,
        no_flatten=False,
    )
