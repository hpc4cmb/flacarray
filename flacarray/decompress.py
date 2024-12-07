# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np

from .libflacarray import decode_flac
from .utils import int32_to_float, keep_select, function_timer, select_keep_indices


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
    use_threads=False,
):
    """Decompress a slice of a FLAC encoded array and restore original data type.

    If `stream_gains` is specified, the output data will be float32 and `stream_offsets`
    must also be provided.  If `stream_gains` is not specified, but `stream_offsets` is,
    then the returned data will be int64.  If neither offsets or gains are specified,
    the decompressed int32 array is returned.

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
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

    Returns:
        (tuple): The (output array, list of stream indices).

    """
    if first_stream_sample is None:
        first_stream_sample = -1
    if last_stream_sample is None:
        last_stream_sample = -1

    starts, nbytes, indices = keep_select(keep, stream_starts, stream_nbytes)
    offsets = select_keep_indices(stream_offsets, indices)
    gains = select_keep_indices(stream_gains, indices)

    if stream_offsets is not None:
        if stream_gains is not None:
            # This is floating point data
            idata = decode_flac(
                compressed,
                starts,
                nbytes,
                stream_size,
                first_sample=first_stream_sample,
                last_sample=last_stream_sample,
                use_threads=use_threads,
            )
            arr = int32_to_float(idata, offsets, gains)
        else:
            # This is int64 data
            idata = decode_flac(
                compressed,
                starts,
                nbytes,
                stream_size,
                first_sample=first_stream_sample,
                last_sample=last_stream_sample,
                use_threads=use_threads,
            )
            ext_shape = offsets.shape + (1,)
            arr = idata.astype(np.int64) + offsets.reshape(ext_shape)
    else:
        if stream_gains is not None:
            raise RuntimeError(
                "When specifying gains, you must also provide the offsets"
            )
        # This is int32 data
        arr = decode_flac(
            compressed,
            starts,
            nbytes,
            stream_size,
            first_sample=first_stream_sample,
            last_sample=last_stream_sample,
            use_threads=use_threads,
        )
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
    use_threads=False,
):
    """Decompress a FLAC encoded array and restore original data type.

    If `stream_gains` is specified, the output data will be float32 and `stream_offsets`
    must also be provided.  If `stream_gains` is not specified, but `stream_offsets` is,
    then the returned data will be int64.  If neither offsets or gains are specified,
    the decompressed int32 array is returned.

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
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

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
        use_threads=use_threads,
    )
    return arr
