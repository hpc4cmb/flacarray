# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .libflacarray import decode_flac
from .utils import (
    int_to_float,
    keep_select,
    function_timer,
    select_keep_indices,
    ensure_one_element,
)


@function_timer
def array_decompress_slice(
    compressed,
    stream_size,
    stream_starts,
    stream_nbytes,
    stream_offsets=None,
    stream_gains=None,
    keep=None,
    first_stream_sample=None,
    last_stream_sample=None,
    is_int64=False,
    use_threads=False,
    no_flatten=False,
):
    """Decompress a slice of a FLAC encoded array and restore original data type.

    If both `stream_gains` and `stream_offsets` are specified, the output will be
    floating point data.  If neither is specified, the output will be integer data.
    It is an error to specify only one of those options.

    The compressed byte stream might contain either int32 or int64 data, and the calling
    code is responsible for tracking this.  The `is_int64` parameter should be set to
    True if the byte stream contains 64bit integers.

    To decompress a subset of samples in all streams, specify the `first_stream_sample`
    and `last_stream_sample` values.  None values or negative values disable this
    feature.

    To decompress a subset of streams, pass a boolean array to the `keep` argument.
    This should have the same shape as the `starts` array.  Only streams with a True
    value in the `keep` array will be decompressed.

    If the `keep` array is specified, the output tuple will contain the 2D array of
    streams that were kept, as well as a list of tuples indicating the original array
    indices for each stream in the output.  If the `keep` array is None, the output
    tuple will contain an array with the original N-dimensional leading array shape
    and the trailing number of samples.  The second element of the tuple will be None.

    Args:
        compressed (array):  The array of compressed bytes.
        stream_size (int):  The length of the decompressed final dimension.
        stream_starts (array):  The array of starting bytes in the bytestream.
        stream_nbytes (array):  The array of number of bytes in each stream.
        stream_offsets (array):  The array of offsets, one per stream.
        stream_gains (array):  The array of gains, one per stream.
        keep (array):  Bool array of streams to keep in the decompression.
        first_stream_sample (int):  The first sample of every stream to decompress.
        last_stream_sample (int):  The last sample of every stream to decompress.
        is_int64 (bool):  If True, the compressed stream contains 64bit integers.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.
        no_flatten (bool):  If True, for single-stream arrays, leave the leading
            dimension of (1,) in the result.

    Returns:
        (tuple): The (output array, list of stream indices).

    """
    if first_stream_sample is None:
        first_stream_sample = -1
    if last_stream_sample is None:
        last_stream_sample = -1

    # If we have one stream, ensure that our auxiliary data are arrays
    is_scalar = False
    if (
        not isinstance(stream_starts, np.ndarray) or
        (len(stream_starts.shape) == 1 and stream_starts.shape[0] == 1)
    ):
        # This is a scalar
        is_scalar = True
        stream_starts = ensure_one_element(stream_starts, np.int64)
        stream_nbytes = ensure_one_element(stream_nbytes, np.int64)
        if stream_offsets is not None:
            # We have float values
            if is_int64:
                stream_offsets = ensure_one_element(stream_offsets, np.float64)
                stream_gains = ensure_one_element(stream_gains, np.float64)
            else:
                stream_offsets = ensure_one_element(stream_offsets, np.float32)
                stream_gains = ensure_one_element(stream_gains, np.float32)

    starts, nbytes, indices = keep_select(keep, stream_starts, stream_nbytes)
    offsets = select_keep_indices(stream_offsets, indices)
    gains = select_keep_indices(stream_gains, indices)

    if stream_offsets is not None:
        if stream_gains is not None:
            # This is floating point data.
            idata = decode_flac(
                compressed,
                starts,
                nbytes,
                stream_size,
                first_sample=first_stream_sample,
                last_sample=last_stream_sample,
                use_threads=use_threads,
                is_int64=is_int64,
            )
            arr = int_to_float(idata, offsets, gains)
        else:
            raise RuntimeError(
                "When specifying offsets, you must also provide the gains"
            )
    else:
        if stream_gains is not None:
            raise RuntimeError(
                "When specifying gains, you must also provide the offsets"
            )
        # This is integer data
        arr = decode_flac(
            compressed,
            starts,
            nbytes,
            stream_size,
            first_sample=first_stream_sample,
            last_sample=last_stream_sample,
            use_threads=use_threads,
            is_int64=is_int64,
        )
    if is_scalar and not no_flatten:
        return (arr.reshape((-1)), indices)
    else:
        return (arr, indices)


@function_timer
def array_decompress(
    compressed,
    stream_size,
    stream_starts,
    stream_nbytes,
    stream_offsets=None,
    stream_gains=None,
    first_stream_sample=None,
    last_stream_sample=None,
    is_int64=False,
    use_threads=False,
    no_flatten=False,
):
    """Decompress a FLAC encoded array and restore original data type.

    If both `stream_gains` and `stream_offsets` are specified, the output will be
    floating point data.  If neither is specified, the output will be integer data.
    It is an error to specify only one of those options.

    The compressed byte stream might contain either int32 or int64 data, and the calling
    code is responsible for tracking this.  The `is_int64` parameter should be set to
    True if the byte stream contains 64bit integers.

    To decompress a subset of samples in all streams, specify the `first_stream_sample`
    and `last_stream_sample` values.  None values or negative values disable this
    feature.

    Args:
        compressed (array):  The array of compressed bytes.
        stream_size (int):  The length of the decompressed final dimension.
        stream_starts (array):  The array of starting bytes in the bytestream.
        stream_nbytes (array):  The array of number of bytes in each stream.
        stream_offsets (array):  The array of offsets, one per stream.
        stream_gains (array):  The array of gains, one per stream.
        first_stream_sample (int):  The first sample of every stream to decompress.
        last_stream_sample (int):  The last sample of every stream to decompress.
        is_int64 (bool):  If True, the compressed stream contains 64bit integers.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.
        no_flatten (bool):  If True, for single-stream arrays, leave the leading
            dimension of (1,) in the result.

    Returns:
        (array): The output array.

    """
    arr, _ = array_decompress_slice(
        compressed,
        stream_size,
        stream_starts,
        stream_nbytes,
        stream_offsets=stream_offsets,
        stream_gains=stream_gains,
        keep=None,
        first_stream_sample=first_stream_sample,
        last_stream_sample=last_stream_sample,
        is_int64=is_int64,
        use_threads=use_threads,
        no_flatten=no_flatten,
    )
    return arr
