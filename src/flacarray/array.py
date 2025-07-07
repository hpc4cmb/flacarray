# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np

from .compress import array_compress
from .decompress import array_decompress_slice
from .hdf5 import write_compressed as hdf5_write_compressed
from .hdf5 import read_compressed as hdf5_read_compressed
from .mpi import global_bytes, global_array_properties
from .utils import log, compressed_dtype
from .zarr import write_compressed as zarr_write_compressed
from .zarr import read_compressed as zarr_read_compressed


class FlacArray:
    """FLAC compressed array representation.

    This class holds a compressed representation of an N-dimensional array.  The final
    (fastest changing) dimension is the axis along which the data is compressed.  Each
    of the vectors in this last dimension is called a "stream" here.  The leading
    dimensions of the original matrix form an array of these streams.

    Internally, the data is stored as a contiguous concatenation of the bytes from
    these compressed streams.  A separate array contains the starting byte of each
    stream in the overall bytes array.  The shape of the starting array corresponds
    to the shape of the leading, un-compressed dimensions of the original array.

    If the input data is 32bit or 64bit integers, each stream in the array is
    compressed directly with FLAC.

    If the input data is 32bit or 64bit floating point numbers, then you **must**
    specify exactly one of either quanta or precision when calling `from_array()`.  For
    floating point data, the mean of each stream is computed and rounded to the nearest
    whole quanta.  This "offset" per stream is recorded and subtracted from the
    stream.  The offset-subtracted stream data is then rescaled and truncated to
    integers (int32 or int64 depending on the bit width of the input array).  If
    `quanta` is specified, the data is rescaled by 1 / quanta.  The quanta may either
    be a scalar applied to all streams, or an array of values, one per stream.  If
    instead the precision (integer number of decimal places) is specified, this is
    converted to a quanta by dividing the stream RMS by `10^{precision}`.  Similar to
    quanta, the precision may be specified as a single value for all streams, or as an
    array of values, one per stream.

    If you choose a quanta value that is close to machine epsilon (e.g. 1e-7 for 32bit
    or 1e-16 for 64bit), then the compression amount will be negligible but the results
    nearly lossless. Compression of floating point data should not be done blindly and
    you should consider the underlying precision of the data you are working with in
    order to achieve the best compression possible.

    The following rules summarize the data conversion that is performed depending on
    the input type:

    * int32:  No conversion.  Compressed to single channel FLAC bytestream.

    * int64:  No conversion.  Compressed to 2-channel (stereo) FLAC bytestream.

    * float32:  Subtract the offset per stream and scale data based on the quanta value
        or precision (see above).  Then round to nearest 32bit integer.

    * float64:  Subtract the offset per stream and scale data based on the quanta value
        or precision (see above).  Then round to nearest 64bit integer.

    After conversion to integers, each stream's data is separately compressed into a
    sequence of FLAC bytes, which is appended to the bytestream.  The offset in bytes
    for each stream is recorded.

    A FlacArray is only constructed directly when making a copy.  Use the class methods
    to create FlacArrays from numpy arrays or on-disk representations.

    Args:
        other (FlacArray):  Construct a copy of the input FlacArray.

    """

    def __init__(
        self,
        other,
        shape=None,
        global_shape=None,
        compressed=None,
        dtype=None,
        stream_starts=None,
        stream_nbytes=None,
        stream_offsets=None,
        stream_gains=None,
        mpi_comm=None,
        mpi_dist=None,
    ):
        if other is not None:
            # We are copying an existing object, make sure we have an
            # independent copy.
            self._shape = copy.deepcopy(other._shape)
            self._global_shape = copy.deepcopy(other._global_shape)
            self._compressed = copy.deepcopy(other._compressed)
            self._dtype = np.dtype(other._dtype)
            self._stream_starts = copy.deepcopy(other._stream_starts)
            self._stream_nbytes = copy.deepcopy(other._stream_nbytes)
            self._stream_offsets = copy.deepcopy(other._stream_offsets)
            self._stream_gains = copy.deepcopy(other._stream_gains)
            self._mpi_dist = copy.deepcopy(other._mpi_dist)
            # MPI communicators can be limited in number and expensive to create.
            self._mpi_comm = other._mpi_comm
        else:
            # This form of constructor is used in the class methods where we
            # have already created these arrays for use by this instance.
            self._shape = shape
            self._global_shape = global_shape
            self._compressed = compressed
            self._dtype = np.dtype(dtype)
            self._stream_starts = stream_starts
            self._stream_nbytes = stream_nbytes
            self._stream_offsets = stream_offsets
            self._stream_gains = stream_gains
            self._mpi_comm = mpi_comm
            self._mpi_dist = mpi_dist
        self._init_params()

    def _init_params(self):
        # The input `_shape` parameter is the original shape when the instance
        # was created from an array or read from disk.  In the case of a single
        # stream, this tracks the user intentions about whether to flatten the
        # leading dimension.  We also track the "local shape", with is the same,
        # but which always keeps the leading dimension.
        if len(self._shape) == 1:
            self._flatten_single = True
            self._local_shape = (1, self._shape[0])
        else:
            self._flatten_single = False
            self._local_shape = self._shape

        self._local_nbytes = self._compressed.nbytes
        (
            self._global_nbytes,
            self._global_proc_nbytes,
            self._global_stream_starts,
        ) = global_bytes(self._local_nbytes, self._stream_starts, self._mpi_comm)
        self._leading_shape = self._local_shape[:-1]
        self._global_leading_shape = self._global_shape[:-1]
        self._stream_size = self._local_shape[-1]

        # For reference, record the type string of the original data.
        self._typestr = self._dtype_str(self._dtype)
        # Track whether we have 32bit or 64bit data
        self._is_int64 = self._dtype == np.dtype(np.int64) or self._dtype == np.dtype(
            np.float64
        )

    @staticmethod
    def _dtype_str(dt):
        if dt == np.dtype(np.float64):
            return "float64"
        elif dt == np.dtype(np.float32):
            return "float32"
        elif dt == np.dtype(np.int64):
            return "int64"
        elif dt == np.dtype(np.int32):
            return "int32"
        else:
            msg = f"Unsupported dtype '{dt}'"
            raise RuntimeError(msg)
        return None

    # Shapes of decompressed array

    @property
    def shape(self):
        """The shape of the local, uncompressed array."""
        return self._shape

    @property
    def global_shape(self):
        """The global shape of array across any MPI communicator."""
        return self._global_shape

    @property
    def leading_shape(self):
        """The local shape of leading uncompressed dimensions."""
        return self._leading_shape

    @property
    def global_leading_shape(self):
        """The global shape of leading uncompressed dimensions across all processes."""
        return self._global_leading_shape

    @property
    def stream_size(self):
        """The uncompressed length of each stream."""
        return self._stream_size

    # Properties of the compressed data

    @property
    def nbytes(self):
        """The total number of bytes used by compressed data on the local process."""
        return self._local_nbytes

    @property
    def global_nbytes(self):
        """The sum of total bytes used by compressed data across all processes."""
        return self._global_nbytes

    @property
    def global_process_nbytes(self):
        """The bytes used by compressed data on each process."""
        return self._global_proc_bytes

    @property
    def nstreams(self):
        """The number of local streams (product of entries of `leading_shape`)"""
        return self._local_nstreams

    @property
    def global_nstreams(self):
        """Number of global streams (product of entries of `global_leading_shape`)"""
        return self._global_nstreams

    @property
    def compressed(self):
        """The concatenated raw bytes of all streams on the local process."""
        return self._compressed

    @property
    def stream_starts(self):
        """The array of starting bytes for each stream on the local process."""
        return self._stream_starts

    @property
    def stream_nbytes(self):
        """The array of nbytes for each stream on the local process."""
        return self._stream_nbytes

    @property
    def global_stream_starts(self):
        """The array of starting bytes within the global compressed data."""
        return self._global_stream_starts

    @property
    def global_stream_nbytes(self):
        """The array of nbytes within the global compressed data."""
        return self._global_stream_nbytes

    @property
    def stream_offsets(self):
        """The value subtracted from each stream during conversion to int32."""
        return self._stream_offsets

    @property
    def stream_gains(self):
        """The gain factor for each stream during conversion to int32."""
        return self._stream_gains

    @property
    def mpi_comm(self):
        """The MPI communicator over which the array is distributed."""
        return self._mpi_comm

    @property
    def mpi_dist(self):
        """The range of the leading dimension assigned to each MPI process."""
        return self._mpi_dist

    @property
    def dtype(self):
        """The dtype of the uncompressed array."""
        return self._dtype

    @property
    def typestr(self):
        """A string representation of the original data type."""
        return self._typestr

    # __getitem__ slicing / decompression on the fly and associated
    # helper functions.

    def _slice_nelem(self, slc, dim):
        """Get the number of elements in a slice."""
        start, stop, step = slc.indices(dim)
        nslc = (stop - start) // step
        if nslc < 0:
            nslc = 0
        return nslc

    def _keep_view(self, key):
        """Convert leading-shape key to bool array."""
        if len(key) != len(self._leading_shape):
            msg = f"keep_view {key} does not match leading "
            msg += f"dimensions {len(self._leading_shape)}"
            raise ValueError(msg)
        view = np.zeros(self._leading_shape, dtype=bool)
        view[key] = True
        return view

    def _get_full_key(self, key):
        """Process the incoming key so that it covers all dimensions.

        Args:
            key (tuple):  The input key consisting of an integer or a tuple
                of slices and / or integers.

        Result:
            (tuple):  The full key.

        """
        ndim = len(self._local_shape)
        full_key = list()
        if self._flatten_single:
            # Our array is a single stream with flattened shape.  The user
            # supplied key should only contain the sample axis.
            if isinstance(key, tuple):
                # It better have length == 1...
                if len(key) != 1:
                    msg = f"Slice key {key} is not valid for single, "
                    msg += "flattened stream."
                    raise ValueError(msg)
                full_key = [0, key[0]]
            else:
                # Single element, compress sample dimension
                full_key = [0, key]
        else:
            if isinstance(key, tuple):
                for axis, axkey in enumerate(key):
                    full_key.append(axkey)
            else:
                full_key.append(key)

        if len(full_key) > ndim:
            msg = f"Invalid slice key {key}, too many dimensions"
            raise ValueError(msg)

        # Fill in remaining dimensions
        filled = len(full_key)
        full_key.extend([slice(None) for x in range(len(self._local_shape) - filled)])
        return full_key

    def _get_leading_axes(self, full_key):
        """Process the leading axes.

        Args:
            full_key (tuple):  The full-rank selection key.

        Returns:
            (tuple):  The (leading_shape, keep array).

        """
        leading_shape = list()
        keep_slice = list()

        if self._flatten_single:
            # Our array is a single stream with flattened shape.
            keep_slice = [0,]
        else:
            for axis, axkey in enumerate(full_key[:-1]):
                if not isinstance(axkey, (int, np.integer)):
                    # Some kind of slice, do not compress this dimension.
                    nslc = self._slice_nelem(axkey, self._local_shape[axis])
                    leading_shape.append(nslc)
                else:
                    # Check for validity
                    if axkey < 0 or axkey >= self._local_shape[axis]:
                        # Insert a zero-length dimension so that a zero-length
                        # array is returned in the calling code.
                        leading_shape.append(0)
                    else:
                        # This dimension is a single element and will be
                        # compressed.
                        pass
                keep_slice.append(axkey)
        leading_shape = tuple(leading_shape)
        keep_slice = tuple(keep_slice)
        if len(keep_slice) == 0:
            keep = None
        else:
            keep = self._keep_view(keep_slice)
        return leading_shape, keep

    def _get_sample_axis(self, full_key):
        """Process any slicing of the stream axis.

        Args:
            full_key (tuple):  The full-rank selection key.

        Returns:
            (tuple):  The (first, last, sample_shape).

        """
        sample_key = full_key[-1]
        if sample_key is None:
            return (0, self._stream_size, (self._stream_size,))
        if isinstance(sample_key, slice):
            start, stop, step = sample_key.indices(self._stream_size)
            if step != 1:
                msg = "Only stride==1 supported on stream slices"
                raise ValueError(msg)
            if stop - start <= 0:
                # No samples
                return (0, 0, (0,))
            return (start, stop, (stop-start,))
        elif isinstance(sample_key, (int, np.integer)):
            # Just a scalar
            return (sample_key, sample_key + 1, ())
        else:
            msg = "Stream dimension supports contiguous slices or single indices."
            raise ValueError(msg)

    def __getitem__(self, raw_key):
        """Decompress a slice of data on the fly.

        Args:
            raw_key (tuple):  A tuple of slices or integers.

        Returns:
            (array):  The decompressed array slice.

        """
        # Get the key for all dimensions
        key = self._get_full_key(raw_key)

        # Compute the output leading shape and keep array
        leading_shape, keep = self._get_leading_axes(key)

        # Compute sample axis slice
        first, last, sample_shape = self._get_sample_axis(key)

        full_shape = leading_shape + sample_shape
        if len(full_shape) == 0:
            n_total = 0
        else:
            n_total = np.prod(full_shape)
        if n_total == 0:
            # At least one dimension was zero, return empty array
            return np.zeros(full_shape, dtype=self._dtype)
        else:
            arr, strm_indices = array_decompress_slice(
                self._compressed,
                self._stream_size,
                self._stream_starts,
                self._stream_nbytes,
                stream_offsets=self._stream_offsets,
                stream_gains=self._stream_gains,
                keep=keep,
                first_stream_sample=first,
                last_stream_sample=last,
                is_int64=self._is_int64,
            )
            return arr.reshape(full_shape)

    def __delitem__(self, key):
        raise RuntimeError("Cannot delete individual streams")

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot modify individual byte streams")

    def __repr__(self):
        rank = 0
        mpistr = ""
        if self._mpi_comm is not None:
            rank = self._mpi_comm.rank
            mpistr = f" | Rank {rank:04d} "
            mpistr += f"{self._mpi_dist[rank][0]}-"
            mpistr += f"{self._mpi_dist[rank][1] - 1} |"
        rep = f"<FlacArray{mpistr} {self._typestr} "
        rep += f"shape={self._shape} bytes={self._local_nbytes}>"
        return rep

    def __eq__(self, other):
        if self._shape != other._shape:
            log.debug(f"other shape {other._shape} != {self._shape}")
            return False
        if self._dtype != other._dtype:
            log.debug(f"other dtype {other._dtype} != {self._dtype}")
            return False
        if self._global_shape != other._global_shape:
            msg = f"other global_shape {other._global_shape} != {self._global_shape}"
            log.debug(msg)
            return False
        if not np.array_equal(self._stream_starts, other._stream_starts):
            msg = f"other starts {other._stream_starts} != {self._stream_starts}"
            log.debug(msg)
            return False
        if not np.array_equal(self._compressed, other._compressed):
            msg = f"other compressed {other._compressed} != {self._compressed}"
            log.debug(msg)
            return False
        if self._stream_offsets is None:
            if other._stream_offsets is not None:
                log.debug("other stream_offsets not None, self is None")
                return False
        else:
            if other._stream_offsets is None:
                log.debug("other stream_offsets is None, self is not None")
                return False
            else:
                if not np.allclose(self._stream_offsets, other._stream_offsets):
                    msg = f"other stream_offsets {other._stream_offsets} != "
                    msg += f"{self._stream_offsets}"
                    log.debug(msg)
                    return False
        if self._stream_gains is None:
            if other._stream_gains is not None:
                log.debug("other stream_gains not None, self is None")
                return False
        else:
            if other._stream_gains is None:
                log.debug("other stream_offsets is None, self is not None")
                return False
            else:
                if not np.allclose(self._stream_gains, other._stream_gains):
                    msg = f"other stream_gains {other._stream_gains} != "
                    msg += f"{self._stream_gains}"
                    log.debug(msg)
                    return False
        return True

    def to_array(
        self,
        keep=None,
        stream_slice=None,
        keep_indices=False,
        use_threads=False,
    ):
        """Decompress local data into a numpy array.

        This uses the compressed representation to reconstruct a normal numpy
        array.  The returned data type will be either int32, int64, float32, or
        float64 depending on the original data type.

        If `stream_slice` is specified, the returned array will have only that
        range of samples in the final dimension.

        If `keep` is specified, this should be a boolean array with the same shape
        as the leading dimensions of the original array.  True values in this array
        indicate that the stream should be kept.

        If `keep` is specified, the returned array WILL NOT have the same shape as
        the original.  Instead it will be a 2D array of decompressed streams- the
        streams corresponding to True values in the `keep` mask.

        If `keep_indices` is True and `keep` is specified, then a tuple of two values
        is returned.  The first is the array of decompressed streams.  The second is
        a list of tuples, each of which specifies the indices of the stream in the
        original array.

        Args:
            keep (array):  Bool array of streams to keep in the decompression.
            stream_slice (slice):  A python slice with step size of one, indicating
                the sample range to extract from each stream.
            keep_indices (bool):  If True, also return the original indices of the
                streams.
            use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
                This is only beneficial for large arrays.

        """
        first_samp = None
        last_samp = None
        if stream_slice is not None:
            if stream_slice.step is not None and stream_slice.step != 1:
                raise RuntimeError(
                    "Only stream slices with a step size of 1 are supported"
                )
            first_samp = stream_slice.start
            last_samp = stream_slice.stop

        arr, indices = array_decompress_slice(
            self._compressed,
            self._stream_size,
            self._stream_starts,
            self._stream_nbytes,
            stream_offsets=self._stream_offsets,
            stream_gains=self._stream_gains,
            keep=keep,
            first_stream_sample=first_samp,
            last_stream_sample=last_samp,
            is_int64=self._is_int64,
            use_threads=use_threads,
            no_flatten=(not self._flatten_single),
        )
        if keep is not None and keep_indices:
            return (arr, indices)
        else:
            return arr

    @classmethod
    def from_array(
        cls, arr, level=5, quanta=None, precision=None, mpi_comm=None, use_threads=False
    ):
        """Construct a FlacArray from a numpy ndarray.

        Args:
            arr (numpy.ndarray):  The input array data.
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
            (FlacArray):  A newly constructed FlacArray.

        """
        # Get the global shape of the array
        global_props = global_array_properties(arr.shape, mpi_comm=mpi_comm)
        global_shape = global_props["shape"]
        mpi_dist = global_props["dist"]

        # Compress our local piece of the array
        compressed, starts, nbytes, offsets, gains = array_compress(
            arr,
            level=level,
            quanta=quanta,
            precision=precision,
            use_threads=use_threads,
        )

        return FlacArray(
            None,
            shape=arr.shape,
            global_shape=global_shape,
            compressed=compressed,
            dtype=arr.dtype,
            stream_starts=starts,
            stream_nbytes=nbytes,
            stream_offsets=offsets,
            stream_gains=gains,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )

    def write_hdf5(self, hgrp):
        """Write data to an HDF5 Group.

        The internal object properties are written to an open HDF5 group.  If you
        wish to use MPI I/O to write data to the group, then you must be using an MPI
        enabled h5py and you should pass in a valid handle to the group on all
        processes.

        If the `FlacArray` is distributed over an MPI communicator, but the h5py
        implementation does not support MPI I/O, then all data will be communicated
        to the rank zero process for writing.  In this case, the `hgrp` argument should
        be None except on the root process.

        Args:
            hgrp (h5py.Group):  The open Group for writing.

        Returns:
            None

        """
        if self._is_int64:
            n_channels = 2
        else:
            n_channels = 1

        hdf5_write_compressed(
            hgrp,
            self._leading_shape,
            self._global_leading_shape,
            self._stream_size,
            self._stream_starts,
            self._global_stream_starts,
            self._stream_nbytes,
            self._stream_offsets,
            self._stream_gains,
            self._compressed,
            n_channels,
            self._compressed.nbytes,
            self._global_nbytes,
            self._global_proc_nbytes,
            self._mpi_comm,
            self._mpi_dist,
        )

    @classmethod
    def read_hdf5(
        cls,
        hgrp,
        keep=None,
        mpi_comm=None,
        mpi_dist=None,
        no_flatten=False,
    ):
        """Construct a FlacArray from an HDF5 Group.

        This function loads all information about the array from an HDF5 group.  If
        `mpi_comm` is specified, the created array is distributed over that
        communicator.  If you also wish to use MPI I/O to read data from the group,
        then you must be using an MPI-enabled h5py and you should pass in a valid
        handle to the group on all processes.

        If `mpi_dist` is specified, it should be an iterable with the number of leading
        dimension elements assigned to each process.  If None, the leading dimension
        will be distributed uniformly.

        If `keep` is specified, this should be a boolean array with the same shape
        as the leading dimensions of the original array.  True values in this array
        indicate that the stream should be kept.

        If `keep` is specified, the returned array WILL NOT have the same shape as
        the original.  Instead it will be a 2D array of decompressed streams- the
        streams corresponding to True values in the `keep` mask.

        Args:
            hgrp (h5py.Group):  The open Group for reading.
            keep (array):  Bool array of streams to keep in the decompression.
            mpi_comm (MPI.Comm):  If specified, the communicator over which to
                distribute the leading dimension.
            mpi_dist (array):  If specified, assign blocks of these sizes to processes
                when distributing the leading dimension.
            no_flatten (bool):  If True, for single-stream arrays, leave the leading
                dimension of (1,) in the result.

        Returns:
            (FlacArray):  A newly constructed FlacArray.

        """
        (
            local_shape,
            global_shape,
            compressed,
            n_channels,
            stream_starts,
            stream_nbytes,
            stream_offsets,
            stream_gains,
            mpi_dist,
            keep_indices,
        ) = hdf5_read_compressed(
            hgrp,
            keep=keep,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )

        dt = compressed_dtype(n_channels, stream_offsets, stream_gains)

        if (len(local_shape) == 2 and local_shape[0] == 1) and not no_flatten:
            # Flatten
            shape = (local_shape[1],)
        else:
            shape = local_shape

        return FlacArray(
            None,
            shape=shape,
            global_shape=global_shape,
            compressed=compressed,
            dtype=dt,
            stream_starts=stream_starts,
            stream_nbytes=stream_nbytes,
            stream_offsets=stream_offsets,
            stream_gains=stream_gains,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )

    def write_zarr(self, zgrp):
        """Write data to an Zarr Group.

        The internal object properties are written to an open zarr group.

        If the `FlacArray` is distributed over an MPI communicator, then all data will
        be communicated to the rank zero process for writing.  In this case, the `zgrp`
        argument should be None except on the root process.

        Args:
            zgrp (zarr.Group):  The open Group for writing.

        Returns:
            None

        """
        if self._is_int64:
            n_channels = 2
        else:
            n_channels = 1
        zarr_write_compressed(
            zgrp,
            self._leading_shape,
            self._global_leading_shape,
            self._stream_size,
            self._stream_starts,
            self._global_stream_starts,
            self._stream_nbytes,
            self._stream_offsets,
            self._stream_gains,
            self._compressed,
            n_channels,
            self._compressed.nbytes,
            self._global_nbytes,
            self._global_proc_nbytes,
            self._mpi_comm,
            self._mpi_dist,
        )

    @classmethod
    def read_zarr(
        cls,
        zgrp,
        keep=None,
        mpi_comm=None,
        mpi_dist=None,
        no_flatten=False,
    ):
        """Construct a FlacArray from a Zarr Group.

        This function loads all information about the array from a zarr group.  If
        `mpi_comm` is specified, the created array is distributed over that
        communicator.

        If `mpi_dist` is specified, it should be an iterable with the number of leading
        dimension elements assigned to each process.  If None, the leading dimension
        will be distributed uniformly.

        If `keep` is specified, this should be a boolean array with the same shape
        as the leading dimensions of the original array.  True values in this array
        indicate that the stream should be kept.

        If `keep` is specified, the returned array WILL NOT have the same shape as
        the original.  Instead it will be a 2D array of decompressed streams- the
        streams corresponding to True values in the `keep` mask.

        Args:
            zgrp (zarr.Group):  The open Group for reading.
            keep (array):  Bool array of streams to keep in the decompression.
            mpi_comm (MPI.Comm):  If specified, the communicator over which to
                distribute the leading dimension.
            mpi_dist (array):  If specified, assign blocks of these sizes to processes
                when distributing the leading dimension.
            no_flatten (bool):  If True, for single-stream arrays, leave the leading
                dimension of (1,) in the result.

        Returns:
            (FlacArray):  A newly constructed FlacArray.

        """
        (
            local_shape,
            global_shape,
            compressed,
            n_channels,
            stream_starts,
            stream_nbytes,
            stream_offsets,
            stream_gains,
            mpi_dist,
            keep_indices,
        ) = zarr_read_compressed(
            zgrp,
            keep=keep,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )

        dt = compressed_dtype(n_channels, stream_offsets, stream_gains)

        if (len(local_shape) == 2 and local_shape[0] == 1) and not no_flatten:
            # Flatten
            shape = (local_shape[1],)
        else:
            shape = local_shape

        return FlacArray(
            None,
            shape=shape,
            global_shape=global_shape,
            compressed=compressed,
            dtype=dt,
            stream_starts=stream_starts,
            stream_nbytes=stream_nbytes,
            stream_offsets=stream_offsets,
            stream_gains=stream_gains,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )
