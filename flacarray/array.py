# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import copy
import logging

import numpy as np

from .compress import array_compress
from .decompress import array_decompress_slice
from .hdf5 import write_compressed as hdf5_write_compressed
from .hdf5 import read_compressed as hdf5_read_compressed
from .mpi import global_bytes, global_array_properties
from .zarr import write_compressed as zarr_write_compressed
from .zarr import read_compressed as zarr_read_compressed

log = logging.getLogger("flacarray")


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

    The input data is converted to 32bit integers.  The "quanta" value is used
    for floating point data conversion and represents the floating point increment
    for a single integer value.  If quanta is None, each stream is scaled independently
    based on its data range.  If quanta is a scalar, all streams are scaled with the
    same value.  If quanta is an array, it specifies the scaling independently for each
    stream.

    Alternatively, if "precision" is provided, each data vector is scaled to retain
    the prescribed number of significant digits when converting to integers.

    The following rules specify the data conversion that is performed depending on
    the input type:

    * int32:  No conversion.

    * int64:  Subtract the integer closest to the mean, then truncate to lower
        32 bits, and check that the higher bits were zero.

    * float32:  Subtract the mean and scale data based on the quanta value (see
        above).  Then round to nearest 32bit integer.

    * float64:  Subtract the mean and scale data based on the quanta value (see
        above).  Then round to nearest 32bit integer.

    After conversion to 32bit integers, each stream's data is separately compressed
    into a sequence of FLAC bytes, which is appended to the bytestream.  The offset in
    bytes for each stream is recorded.

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
            self._stream_starts = stream_starts
            self._stream_nbytes = stream_nbytes
            self._stream_offsets = stream_offsets
            self._stream_gains = stream_gains
            self._mpi_comm = mpi_comm
            self._mpi_dist = mpi_dist
        self._init_params()

    def _init_params(self):
        self._local_nbytes = self._compressed.nbytes
        (
            self._global_nbytes,
            self._global_proc_nbytes,
            self._global_stream_starts,
        ) = global_bytes(self._local_nbytes, self._stream_starts, self._mpi_comm)
        self._stream_size = self._shape[-1]
        self._leading_shape = self._stream_starts.shape
        self._local_nstreams = np.prod(self._leading_shape)
        if len(self._global_shape) == 1:
            self._global_leading_shape = (1,)
        else:
            self._global_leading_shape = self._global_shape[:-1]
        self._global_nstreams = np.prod(self._global_leading_shape)
        # For reference, record the type of the original data.
        if self._stream_offsets is not None:
            if self._stream_gains is not None:
                # This is floating point data
                if self._stream_gains.dtype == np.dtype(np.float64):
                    self._typestr = "float64"
                else:
                    self._typestr = "float32"
            else:
                # This is int64 data
                self._typestr = "int64"
        else:
            self._typestr = "int32"

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
        return self._shape[-1]

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

    def _keep_view(self, key):
        if len(key) != len(self._leading_shape):
            raise ValueError("view size does not match leading dimensions")
        view = np.zeros(self._leading_shape, dtype=bool)
        view[key] = True
        return view

    def __getitem__(self, key):
        """Decompress a slice of data on the fly."""
        first = None
        last = None
        keep = None
        if isinstance(key, tuple):
            # We are slicing on multiple dimensions
            if len(key) == len(self._shape):
                # Slicing on the sample dimension too
                keep = self._keep_view(key[:-1])
                samp_key = key[-1]
                if isinstance(samp_key, slice):
                    # A slice
                    if samp_key.step is not None and samp_key.step != 1:
                        raise ValueError("Only stride==1 supported on stream slices")
                    first = samp_key.start
                    last = samp_key.stop
                elif isinstance(samp_key, (int, np.integer)):
                    # Just a scalar
                    first = samp_key
                    last = samp_key + 1
                else:
                    raise ValueError(
                        "Only contiguous slices supported on the stream dimension"
                    )
            else:
                # Only slicing the leading dimensions
                vw = list(key)
                vw.extend(
                    [slice(None) for x in range(len(self._leading_shape) - len(key))]
                )
                keep = self._keep_view(tuple(vw))
        else:
            # We are slicing / indexing only the leading dimension
            vw = [slice(None) for x in range(len(self._leading_shape))]
            vw[0] = key
            keep = self._keep_view(tuple(vw))

        arr, _ = array_decompress_slice(
            self._compressed,
            self._stream_size,
            self._stream_starts,
            self._stream_nbytes,
            stream_offsets=self._stream_offsets,
            stream_gains=self._stream_gains,
            keep=keep,
            first_stream_sample=first,
            last_stream_sample=last,
        )
        return arr

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
        self, keep=None, stream_slice=None, keep_indices=False, use_threads=False
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
            use_threads=use_threads,
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

        Returns:
            (FlacArray):  A newly constructed FlacArray.

        """
        (
            local_shape,
            global_shape,
            compressed,
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

        return FlacArray(
            None,
            shape=local_shape,
            global_shape=global_shape,
            compressed=compressed,
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

        Returns:
            (FlacArray):  A newly constructed FlacArray.

        """
        (
            local_shape,
            global_shape,
            compressed,
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

        return FlacArray(
            None,
            shape=local_shape,
            global_shape=global_shape,
            compressed=compressed,
            stream_starts=stream_starts,
            stream_nbytes=stream_nbytes,
            stream_offsets=stream_offsets,
            stream_gains=stream_gains,
            mpi_comm=mpi_comm,
            mpi_dist=mpi_dist,
        )
