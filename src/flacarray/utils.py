# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import inspect
import logging
import os
import time
from functools import wraps

import numpy as np

from .libflacarray import (
    wrap_float32_to_int32,
    wrap_float64_to_int64,
    wrap_int32_to_float32,
    wrap_int64_to_float64,
)


log = logging.getLogger()
log.setLevel(logging.INFO)
env_keys = ["FLACARRAY_LOGLEVEL", "FLACARRAY_LOG_LEVEL"]
for env_key in env_keys:
    if env_key in os.environ:
        lvl = os.environ[env_key]
        if hasattr(logging, lvl):
            log.setLevel(getattr(logging, lvl))
        else:
            msg = f"Environment variable {env_key} set to invalid level '{lvl}'"
            raise RuntimeError(msg)


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


def ensure_one_element(input, dtype=None):
    """Helper function to check dimension and dtype.

    If the input is an array, it is verified to be a single element with the
    specified dtype.  If the input is a scalar, it is promoted to an array
    with the specified type.

    Args:
        input (array, scalar): The input value
        dtype (np.dtype): The dtype to check

    Returns:
        (array): The original array or a new array.

    """
    if isinstance(input, np.ndarray):
        # This is an array, should be 1D
        if input.shape != (1,):
            msg = "Input array does not have a single element."
            raise ValueError(msg)
        if dtype is not None:
            if input.dtype != np.dtype(dtype):
                msg = f"Input has dtype {input.dtype}, not {dtype}"
                raise ValueError(msg)
        result = input
    else:
        # Promote scalar to array
        if dtype is None:
            raise ValueError("Input is a scalar, dtype must be specified")
        result = np.array([input], dtype=dtype)
    return result


def compressed_dtype(n_channel, offsets, gains):
    """Helper function to determine dtype of compressed data.

    At several places in the code (for example when reading compressed data),
    we have access to the number of FLAC channels and the offset and gain arrays.
    This function returns the corresponding dtype.

    Args:
        n_channel (int):  The number of FLAC channels.
        offsets (array):  The offsets or None.
        gains (array):  The gains or None.

    Returns:
        (dtype):  The dtype of the decompressed data.

    """
    if n_channel == 2:
        # 64bit
        if offsets is None or gains is None:
            # integer
            return np.dtype(np.int64)
        else:
            # float
            return np.dtype(np.float64)
    else:
        # 32bit
        if offsets is None or gains is None:
            # integer
            return np.dtype(np.int32)
        else:
            # float
            return np.dtype(np.float32)


@function_timer
def float_to_int(data, quanta=None, precision=None):
    """Convert floating point data to integers.

    This function subtracts the mean and rescales data before rounding to 32bit
    or 64bit integer values.  32bit floats are converted to 32bit integers and
    64bit floats are converted to 64bit integers.

    See discussion in the `FlacArray` class documentation about how the offsets and
    gains are computed for a given quanta.

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
            # This worked, it is an array.  Check shape
            if precision.shape != leading_shape:
                msg = f"precision array ({precision}) has shape that does not "
                msg += f"match leading shape of data ({precision.shape} != "
                msg += f"{leading_shape})"
                raise RuntimeError(msg)
            quanta = rms.reshape(leading_shape) / 10 ** precision.reshape(leading_shape)
        except TypeError:
            # Precision is a scalar
            quanta = rms.reshape(leading_shape) / 10**precision

    if quanta is None:
        # Indicate this by passing a fake value
        quanta = np.zeros(0, dtype=data.dtype)
    else:
        # Make sure it is an array
        try:
            lquant = len(quanta)
            # Worked. Check shape
            if quanta.shape != leading_shape:
                msg = f"quanta array ({quanta}) has shape that does not "
                msg += f"match leading shape of data ({quanta.shape} != "
                msg += f"{leading_shape})"
                raise RuntimeError(msg)
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
        output, offsets, gains = wrap_float64_to_int64(
            data.reshape((-1,)),
            n_stream,
            stream_size,
            quanta.reshape((-1,)).astype(data.dtype),
        )

    if len(leading_shape) == 0:
        # Single input stream
        return (
            output.reshape(data.shape),
            offsets.reshape((-1,)),
            gains.reshape((-1,)),
        )
    else:
        # Reshape flat arrays to the leading shape
        return (
            output.reshape(data.shape),
            offsets.reshape(leading_shape),
            gains.reshape(leading_shape),
        )


@function_timer
def int_to_float(idata, offset, gain):
    """Restore floating point data from integers.

    The gain and offset are applied and the resulting data is returned.
    32bit integer data is converted to 32bit floats and 64bit integer data
    is converted to 64bit floats.

    Args:
        idata (array):  The 32bit or 64bit integer data.
        offset (array):  The offset used in the original conversion.
        gain (array):  The gain used in the original conversion.

    Returns:
        (array):  The restored float data.

    """
    if idata.dtype != np.dtype(np.int32) and idata.dtype != np.dtype(np.int64):
        raise ValueError("Input data should be int32 or int64")
    if idata.dtype == np.dtype(np.int32):
        is_int64 = False
    else:
        is_int64 = True

    leading_shape = idata.shape[:-1]
    if len(leading_shape) == 0 or (len(leading_shape) == 1 and leading_shape[0] == 1):
        # Promote scalar values if needed.
        n_stream = 1
        if is_int64:
            offset = ensure_one_element(offset, np.float64)
            gain = ensure_one_element(gain, np.float64)
        else:
            offset = ensure_one_element(offset, np.float32)
            gain = ensure_one_element(gain, np.float32)
    else:
        n_stream = np.prod(leading_shape)
        if offset.shape != leading_shape:
            msg = f"Offset array has shape {offset.shape}, expected "
            msg += f"shape {leading_shape}"
            raise ValueError(msg)
        if gain.shape != leading_shape:
            msg = f"Gain array has shape {gain.shape}, expected "
            msg += f"shape {leading_shape}"
            raise ValueError(msg)
    stream_size = idata.shape[-1]

    if idata.dtype == np.dtype(np.int32):
        result = wrap_int32_to_float32(
            idata.reshape((-1,)),
            n_stream,
            stream_size,
            offset.reshape((-1,)),
            gain.reshape((-1,)),
        )
    else:
        result = wrap_int64_to_float64(
            idata.reshape((-1,)),
            n_stream,
            stream_size,
            offset.reshape((-1,)),
            gain.reshape((-1,)),
        )
    # The C code returns a flat-packed array of streams
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
    it = np.nditer(keep, order="C", flags=["multi_index"])
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
