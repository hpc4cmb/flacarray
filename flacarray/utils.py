# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import inspect
import os
import time
from functools import wraps

import numpy as np

from .libflacarray import (
    wrap_int64_to_int32,
    wrap_float32_to_int32,
    wrap_float64_to_int32,
    wrap_int32_to_int64,
    wrap_int32_to_float32,
    wrap_int32_to_float64,
)


_use_function_timers = None
_function_timer_env_var = "FLACARRAY_TIMING"


def use_function_timers():
    global _use_function_timers
    if _use_function_timers is not None:
        # Already checked
        return _use_function_timers

    if _function_timer_env_var in os.environ:
        valstr = os.environ[_function_timer_env_var]
        if valstr == "1" or valstr == "true" or valstr == "yes":
            _use_function_timers = True
        else:
            _use_function_timers = False
    else:
        _use_function_timers = False
    return _use_function_timers


_global_timers = None


def get_timers():
    global _global_timers
    if _global_timers is None:
        _global_timers = dict()
    return _global_timers


def update_timer(name, elapsed):
    tmrs = get_timers()
    if name not in tmrs:
        tmrs[name] = 0.0
    tmrs[name] += elapsed


def clear_timers():
    global _global_timers
    if _global_timers is None:
        _global_timers = dict()
    _global_timers.clear()


def print_timers():
    timers = get_timers()
    for k, v in timers.items():
        print(f"{k}:  {v} seconds", flush=True)


# Global list of functions to ignore in our simplified timing stacktrace.
# This can be updated by decorating functions with the function_timer_stackskip
# below.

_timing_stack_skip = {
    "df",
    "<module>",
}


def function_timer(f):
    """Simple decorator for function timing.

    If the FLACARRAY_TIMING environment variable is set, enable function timers
    within the package.

    """
    if use_function_timers():
        fname = f"{f.__qualname__}"

        @wraps(f)
        def df(*args, **kwargs):
            global _timing_stack_skip
            # Build a name from the current function and the call trace.
            tnm = ""
            fnmlist = list()
            frm = inspect.currentframe().f_back
            while frm:
                if "self" in frm.f_locals:
                    # this is inside a class instance
                    funcname = f"{frm.f_locals['self'].__class__.__name__}.{frm.f_code.co_name}"
                    if funcname not in _timing_stack_skip:
                        found = False
                        for base in frm.f_locals["self"].__class__.__bases__:
                            basename = f"{base.__name__}.{frm.f_code.co_name}"
                            if basename in _timing_stack_skip:
                                found = True
                                break
                        if not found:
                            fnmlist.append(funcname)
                else:
                    # this is just a function call
                    if frm.f_code.co_name not in _timing_stack_skip:
                        fnmlist.append(frm.f_code.co_name)
                frm = frm.f_back

            if len(fnmlist) > 0:
                tnm += "|".join(reversed(fnmlist))
                tnm += "|"

            # Make sure the final frame handle is released
            del frm
            tnm += fname
            start = time.perf_counter()
            result = f(*args, **kwargs)
            stop = time.perf_counter()
            elapsed = stop - start
            update_timer(tnm, elapsed)
            return result

    else:

        @wraps(f)
        def df(*args, **kwargs):
            return f(*args, **kwargs)

    return df


def function_timer_stackskip(f):
    if use_function_timers():

        @wraps(f)
        def df(*args, **kwargs):
            global _timing_stack_skip
            funcname = None
            if inspect.ismethod(f):
                funcname = f.__self__.__name__
            else:
                funcname = f.__qualname__
            if funcname not in _timing_stack_skip:
                _timing_stack_skip.add(funcname)
            return f(*args, **kwargs)

    else:

        @wraps(f)
        def df(*args, **kwargs):
            return f(*args, **kwargs)

    return df


@function_timer
def int64_to_int32(data):
    """Convert an array of 64bit integer streams to 32bit.

    For each stream, this finds the 64bit integer mean and subtracts it.  It then
    checks that the stream values will fit in a 32bit integer representation.  If you
    want to treat the integer values as floating point data, use float_to_int32
    instead.

    The offset array returned will have the same shape as the leading dimensions of
    the input array.

    Args:
        data (array):  The 64bit integer array.

    Returns:
        (tuple):  The (integer data, offset array)

    """
    if data.dtype != np.dtype(np.int64):
        raise ValueError("Only int64 data is supported by this function")

    leading_shape = data.shape[:-1]
    if len(leading_shape) == 0:
        n_stream = 1
    else:
        n_stream = np.prod(leading_shape)
    stream_size = data.shape[-1]

    output, offsets = wrap_int64_to_int32(
        data.reshape((-1,)),
        n_stream,
        stream_size,
    )

    return (
        output.reshape(data.shape),
        offsets.reshape(leading_shape),
    )


@function_timer
def float_to_int32(data, quanta=None, precision=None):
    """Convert floating point data to integers.

    This function subtracts the mean and rescales data before rounding to 32bit
    integer values.

    Args:
        data (array):  The floating point data.
        quanta (float):  The floating point quantity corresponding to one integer
            resolution amount in the output.  If `None`, quanta will be
            based on the full dynamic range of the data.
        precision (int):  Number of significant digits to preserve.  If
            provided, `quanta` will be estimated accordingly.

    Returns:
        (tuple):  The (integer data, offset array, gain array)

    """
    if np.any(np.isnan(data)):
        raise RuntimeError("Cannot convert data with NaNs to integers")
    if quanta is not None and precision is not None:
        raise RuntimeError("Cannot specify both quanta and precision")
    if data.dtype != np.dtype(np.float32) and data.dtype != np.dtype(np.float64):
        raise ValueError("Only float32 and float64 data are supported")

    leading_shape = data.shape[:-1]
    if len(leading_shape) == 0:
        n_stream = 1
    else:
        n_stream = np.prod(leading_shape)
    stream_size = data.shape[-1]

    if precision is not None:
        # Convert precision into quanta array
        rms = np.std(data, axis=-1, keepdims=True)
        try:
            lprec = len(precision)
            # This worked, it is an array
            quanta = rms / 10 ** precision.reshape(leading_shape + (1,))
        except TypeError:
            # Precision is a scalar
            quanta = rms / 10**precision

    if quanta is None:
        # Indicate this by passing a fake value
        quanta = np.zeros(0, dtype=data.dtype)
    else:
        # Make sure it is an array
        try:
            lquant = len(quanta)
            # Worked...
        except TypeError:
            quanta = quanta * np.ones(leading_shape, dtype=data.dtype)

    if data.dtype == np.dtype(np.float32):
        output, offsets, gains = wrap_float32_to_int32(
            data.reshape((-1,)),
            n_stream,
            stream_size,
            quanta.reshape((-1,)).astype(data.dtype),
        )
    else:
        output, offsets, gains = wrap_float64_to_int32(
            data.reshape((-1,)),
            n_stream,
            stream_size,
            quanta.reshape((-1,)).astype(data.dtype),
        )

    return (
        output.reshape(data.shape),
        offsets.reshape(leading_shape),
        gains.reshape(leading_shape),
    )


@function_timer
def int32_to_float(idata, offset, gain):
    """Restore floating point data from integers.

    The gain and offset are applied and the resulting data is returned.

    Args:
        idata (array):  The 32bit integer data.
        offset (array):  The offset used in the original conversion.
        gain (array):  The gain used in the original conversion.

    Returns:
        (array):  The restored float data.

    """
    if idata.dtype != np.dtype(np.int32):
        raise ValueError("Input data should be int32")

    leading_shape = idata.shape[:-1]
    if len(leading_shape) == 0:
        n_stream = 1
    else:
        n_stream = np.prod(leading_shape)
    stream_size = idata.shape[-1]

    if offset.shape != leading_shape:
        msg = f"Offset array has shape {offset.shape}, expected shape {leading_shape}"
        raise ValueError(msg)

    if gain.shape != leading_shape:
        msg = f"Gain array has shape {gain.shape}, expected shape {leading_shape}"
        raise ValueError(msg)

    if gain.dtype == np.dtype(np.float32):
        result = wrap_int32_to_float32(
            idata.reshape((-1,)),
            n_stream,
            stream_size,
            offset.reshape((-1,)),
            gain.reshape((-1,)),
        )
    else:
        result = wrap_int32_to_float64(
            idata.reshape((-1,)),
            n_stream,
            stream_size,
            offset.reshape((-1,)),
            gain.reshape((-1,)),
        )
    return result.reshape(idata.shape)


def keep_select(keep, stream_starts, stream_nbytes):
    """Filter out a subset of streams.

    Given a keep mask, return the selected stream starts / nbytes as well as the
    array of selected indices.

    Args:
        keep (array):  Bool array of streams to keep in the decompression.
        stream_starts (array):  The array of starting bytes in the bytestream.
        stream_nbytes (array):  The array of number of bytes in each stream.

    Returns:
        (tuple):  The new (stream starts, stream nbytes, indices).

    """
    if keep is None:
        return (stream_starts, stream_nbytes, None)
    if keep.shape != stream_starts.shape:
        raise RuntimeError("The keep array should have the same shape as stream_starts")
    if keep.shape != stream_nbytes.shape:
        raise RuntimeError("The keep array should have the same shape as stream_starts")
    starts = list()
    nbytes = list()
    indices = list()
    it = np.nditer(keep, flags=["multi_index"])
    for st in it:
        idx = it.multi_index
        if st:
            # We are keeping this stream
            starts.append(stream_starts[idx])
            nbytes.append(stream_nbytes[idx])
            indices.append(idx)
    it.close()
    del it
    return (
        np.array(starts, dtype=np.int64),
        np.array(nbytes, dtype=np.int64),
        indices,
    )


def select_keep_indices(arr, indices):
    """Helper function to extract array elements with a list of indices."""
    if arr is None:
        return None
    if indices is None:
        return arr
    dt = arr.dtype
    return np.array([arr[x] for x in indices], dtype=dt)
