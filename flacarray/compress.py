# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .libflacarray import encode_flac
from .utils import int64_to_int32, float_to_int32, function_timer


@function_timer
def array_compress(arr, level=5, quanta=None, precision=None, use_threads=False):
    """Compress a numpy array with optional floating point conversion.

    If `arr` is an int32 array, the returned stream offsets and gains will be None.
    if `arr` is an int64 array, the stream offsets will be the integer value subtracted
    when converting to int32.  Both float32 and float64 data will have floating point
    offset and gain arrays returned.

    Args:
        arr (numpy.ndarray):  The input array data.
        level (int):  Compression level (0-8).
        quanta (float, array):  For floating point data, the floating point
            increment of each 32bit integer value.  Optionally an array of
            increments, one per stream.
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
        dquanta = None

    if arr.dtype == np.dtype(np.int32):
        (compressed, starts, nbytes) = encode_flac(arr, level, use_threads=use_threads)
        return (compressed, starts, nbytes, None, None)
    elif arr.dtype == np.dtype(np.int64):
        idata, ioff = int64_to_int32(arr)
        (compressed, starts, nbytes) = encode_flac(
            idata, level, use_threads=use_threads
        )
        return (compressed, starts, nbytes, ioff, None)
    elif arr.dtype == np.dtype(np.float64) or arr.dtype == np.dtype(np.float32):
        idata, foff, gains = float_to_int32(arr, quanta=dquanta, precision=precision)
        (compressed, starts, nbytes) = encode_flac(
            idata, level, use_threads=use_threads
        )
        return (compressed, starts, nbytes, foff, gains)
    else:
        raise ValueError(f"Unsupported data type '{arr.dtype}'")
