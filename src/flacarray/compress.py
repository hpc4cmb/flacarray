# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from ._libflacarray import encode_flac
from .utils import float_to_int, function_timer


@function_timer
def array_compress(arr, level=5, quanta=None, precision=None, use_threads=False):
    """Compress a numpy array with optional floating point conversion.

    If `arr` is an int32 array, the returned stream offsets and gains will be None.
    if `arr` is an int64 array, the returned stream offsets and gains will be None and
    the calling code is responsible for tracking that the compressed bytes are
    associated with a 64bit stream.

    If the input array is float32 or float64, exactly one of quanta or precision
    must be specified.  Both float32 and float64 data will have floating point offset
    and gain arrays returned.  See discussion in the `FlacArray` class documentation
    about how the offsets and gains are computed for a given quanta.

    The shape of the returned auxiliary arrays (starts, nbytes, etc) will have a shape
    corresponding to the leading shape of the input array.  If the input array is a
    single stream, the returned auxiliary information will be arrays with a single
    element.

    Args:
        arr (numpy.ndarray):  The input array data.
        level (int):  Compression level (0-8).
        quanta (float, array):  For floating point data, the floating point
            increment of each integer value.  Optionally an array of increments,
            one per stream.
        precision (int, array):  Number of significant digits to retain in
            float-to-int conversion.  Alternative to `quanta`.  Optionally an
            iterable of values, one per stream.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

    Returns:
        (tuple): The (compressed bytes, stream starts, stream_nbytes, stream offsets,
            stream gains)

    """
    if arr.size == 0:
        raise ValueError("Cannot compress a zero-sized array!")
    leading_shape = arr.shape[:-1]

    if arr.dtype == np.dtype(np.float32) or arr.dtype == np.dtype(np.float64):
        # Floating point data
        if quanta is None and precision is None:
            msg = f"Compressing floating point data ('{arr.dtype}') "
            msg = "requires specifying either quanta or precision."
            raise RuntimeError(msg)
        if quanta is not None:
            if precision is not None:
                raise RuntimeError("Cannot set both quanta and precision")
            try:
                nq = len(quanta)
                # This is an array
                if nq.shape != leading_shape:
                    msg = "If not a scalar, quanta must have the same shape as the "
                    msg += "leading dimensions of the array"
                    raise ValueError(msg)
                dquanta = quanta.astype(arr.dtype)
            except TypeError:
                # This is a scalar, applied to all detectors
                dquanta = quanta * np.ones(leading_shape, dtype=arr.dtype)
        else:
            # We are using precision instead
            dquanta = None
        idata, foff, gains = float_to_int(arr, quanta=dquanta, precision=precision)
        (compressed, starts, nbytes) = encode_flac(
            idata, level, use_threads=use_threads
        )
        return (compressed, starts, nbytes, foff, gains)
    elif arr.dtype == np.dtype(np.int32) or arr.dtype == np.dtype(np.int64):
        # Integer data
        (compressed, starts, nbytes) = encode_flac(arr, level, use_threads=use_threads)
        return (compressed, starts, nbytes, None, None)
    else:
        raise ValueError(f"Unsupported data type '{arr.dtype}'")


def array_compress_empty(global_shape, dtype, quanta, precision):
    """Mock the compressed data parameters for processes with no data.

    When using MPI, some processes may have no local data.  This helper function
    returns the equivalent parameters for those processes which cannot call
    `array_compress()`.

    Args:
        global_shape (tuple):  The shape of the global data
        dtype (np.dtype):  The array dtype
        quanta (array):  The quanta values (only checked for None)
        precision (array):  The precision (only check for None)

    Returns:
        (tuple): The (compressed bytes, stream starts, stream_nbytes, stream offsets,
            stream gains)

    """
    leading_shape = global_shape[:-1]
    compressed = np.zeros(0, dtype=np.uint8)
    if len(leading_shape[1:]) == 0:
        starts_shape = (0,)
    else:
        starts_shape = (0,) + leading_shape[1:]
    starts = np.zeros(starts_shape, dtype=np.int64)
    nbytes = np.zeros(starts_shape, dtype=np.int64)
    if quanta is None and precision is None:
        offsets = None
        gains = None
    else:
        offsets = np.zeros(starts_shape, dtype=dtype)
        gains = np.zeros(starts_shape, dtype=dtype)
    return (compressed, starts, nbytes, offsets, gains)
