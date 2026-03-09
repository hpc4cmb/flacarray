
from libc.stdint cimport uint32_t, int32_t, int64_t
from cpython cimport bool

from cython.view cimport array as cvarray

import numpy as np
cimport numpy as cnp

cnp.import_array()

flac_i32_dtype = np.dtype(np.int32)
flac_i64_dtype = np.dtype(np.int64)
compressed_dtype = np.dtype(np.uint8)
offset_dtype = np.dtype(np.int64)


cdef extern from "flacarray.h":
    int encode_i32(
        int32_t * data,
        int64_t n_stream,
        int64_t stream_size,
        uint32_t level,
        int64_t * n_bytes,
        int64_t * starts,
        unsigned char ** rawbytes
    )
    int encode_i32_threaded(
        int32_t * data,
        int64_t n_stream,
        int64_t stream_size,
        uint32_t level,
        int64_t * n_bytes,
        int64_t * starts,
        unsigned char ** rawbytes
    )
    int encode_i64(
        int64_t * data,
        int64_t n_stream,
        int64_t stream_size,
        uint32_t level,
        int64_t * n_bytes,
        int64_t * starts,
        unsigned char ** rawbytes
    )
    int encode_i64_threaded(
        int64_t * data,
        int64_t n_stream,
        int64_t stream_size,
        uint32_t level,
        int64_t * n_bytes,
        int64_t * starts,
        unsigned char ** rawbytes
    )
    int decode_i32(
        unsigned char * rawbytes,
        int64_t * starts,
        int64_t * nbytes,
        int64_t n_stream,
        int64_t stream_size,
        int64_t first_sample,
        int64_t last_sample,
        int32_t * data,
        bool use_threads
    )
    int decode_i64(
        unsigned char * rawbytes,
        int64_t * starts,
        int64_t * nbytes,
        int64_t n_stream,
        int64_t stream_size,
        int64_t first_sample,
        int64_t last_sample,
        int64_t * data,
        bool use_threads
    )
    int float32_to_int32(
        float * input,
        int64_t n_stream,
        int64_t stream_size,
        float * quanta,
        int32_t * output,
        float * offsets,
        float * gains
    )
    int float64_to_int64(
        double * input,
        int64_t n_stream,
        int64_t stream_size,
        double * quanta,
        int64_t * output,
        double * offsets,
        double * gains
    )
    void int64_to_float64(
        int64_t * input,
        int64_t n_stream,
        int64_t stream_size,
        double * offsets,
        double * gains,
        double * output
    )
    void int32_to_float32(
        int32_t * input,
        int64_t n_stream,
        int64_t stream_size,
        float * offsets,
        float * gains,
        float * output
    )


def wrap_float32_to_int32(
    cnp.ndarray[float, ndim=1, mode="c"] flatdata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.ndarray[float, ndim=1, mode="c"] quanta,
):
    """Convert an array of 32bit float streams to 32bit integers.

    This function subtracts the mean and rescales data before rounding to 32bit
    integer values.

    Args:
        flatdata (array):  The 32bit float array.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        quanta (array):  Array of values for each stream.  If the length does not
            equal the number of streams, then it will be ignored and computed from
            the data range.

    Returns:
        (tuple):  The (integer data, offset array, gain array)

    """
    # Allocate the outputs
    cdef int64_t size = n_stream * stream_size
    cdef cnp.ndarray output = np.empty(size, dtype=np.int32, order="C")
    cdef cnp.ndarray offsets = np.empty(n_stream, dtype=np.float32, order="C")
    cdef cnp.ndarray gains = np.empty(n_stream, dtype=np.float32, order="C")

    cdef float * fquanta = NULL
    if len(quanta) == n_stream:
        fquanta = <float *>quanta.data

    errcode = float32_to_int32(
        <float *>flatdata.data,
        n_stream,
        stream_size,
        fquanta,
        <cnp.int32_t *>output.data,
        <float *>offsets.data,
        <float *>gains.data,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Encoding failed, return code = {errcode}"
        raise RuntimeError(msg)

    return (output, offsets, gains)


def wrap_float64_to_int64(
    cnp.ndarray[double, ndim=1, mode="c"] flatdata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.ndarray[double, ndim=1, mode="c"] quanta,
):
    """Convert an array of 64bit float streams to 64bit integers.

    This function subtracts the mean and rescales data before rounding to 32bit
    integer values.

    Args:
        flatdata (array):  The 64bit float array.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        quanta (array):  Array of values for each stream.  If the length does not
            equal the number of streams, then it will be ignored and computed from
            the data range.

    Returns:
        (tuple):  The (integer data, offset array, gain array)

    """
    # Allocate the outputs
    cdef int64_t size = n_stream * stream_size
    cdef cnp.ndarray output = np.empty(size, dtype=np.int64, order="C")
    cdef cnp.ndarray offsets = np.empty(n_stream, dtype=np.float64, order="C")
    cdef cnp.ndarray gains = np.empty(n_stream, dtype=np.float64, order="C")

    cdef double * fquanta = NULL
    if len(quanta) == n_stream:
        fquanta = <double *>quanta.data

    errcode = float64_to_int64(
        <double *>flatdata.data,
        n_stream,
        stream_size,
        fquanta,
        <cnp.int64_t *>output.data,
        <double *>offsets.data,
        <double *>gains.data,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Encoding failed, return code = {errcode}"
        raise RuntimeError(msg)

    return (output, offsets, gains)


def wrap_int32_to_float32(
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] idata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.ndarray[float, ndim=1, mode="c"] offsets,
    cnp.ndarray[float, ndim=1, mode="c"] gains,
):
    """Restore int32 data to float32.

    Args:
        idata (array):  The 32bit integer array.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        offsets (array):  The stream offsets.
        gains (array):  The stream gains.

    Returns:
        (array):  The output data.

    """
    # Allocate the outputs
    cdef int64_t size = n_stream * stream_size
    cdef cnp.ndarray output = np.empty(size, dtype=np.float32, order="C")

    int32_to_float32(
        <cnp.int32_t *>idata.data,
        n_stream,
        stream_size,
        <float *>offsets.data,
        <float *>gains.data,
        <float *>output.data,
    )
    return output


def wrap_int64_to_float64(
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] idata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.ndarray[double, ndim=1, mode="c"] offsets,
    cnp.ndarray[double, ndim=1, mode="c"] gains,
):
    """Restore int64 data to float64.

    Args:
        idata (array):  The 64bit integer array.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        offsets (array):  The stream offsets.
        gains (array):  The stream gains.

    Returns:
        (array):  The output data.

    """
    # Allocate the outputs
    cdef int64_t size = n_stream * stream_size
    cdef cnp.ndarray output = np.empty(size, dtype=np.float64, order="C")

    int64_to_float64(
        <cnp.int64_t *>idata.data,
        n_stream,
        stream_size,
        <double *>offsets.data,
        <double *>gains.data,
        <double *>output.data,
    )
    return output


def wrap_encode_i32(
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] flatdata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.uint32_t level,
):
    """Wrapper around the C int32 encode function.

    This does some data setup, but works with a flat-packed version of
    the input array.

    Args:
        flatdata (array):  The 1D reshaped view of the data.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        level (uint32_t):  The compression level (0-8).

    Returns:
        (tuple): The (compressed bytes, flat-packed starting bytes,
            flat-packed stream bytes).

    """
    # Allocate the array of starts
    cdef cnp.ndarray flat_starts = np.empty(n_stream, dtype=np.int64, order="C")
    cdef cnp.ndarray flat_nbytes = np.empty(n_stream, dtype=np.int64, order="C")

    cdef int64_t n_bytes
    cdef unsigned char * rawbytes
    cdef int errcode = 0

    errcode = encode_i32(
        <cnp.int32_t *>flatdata.data,
        n_stream,
        stream_size,
        level,
        &n_bytes,
        <cnp.int64_t *>flat_starts.data,
        &rawbytes,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Encoding failed, return code = {errcode}"
        raise RuntimeError(msg)

    # Compute the bytes per stream
    flat_nbytes[:-1] = np.diff(flat_starts)
    flat_nbytes[-1] = n_bytes - flat_starts[-1]

    # Wrap the returned C-allocated buffers so that they are properly garbage
    # collected.
    cdef cvarray compressed = <cnp.uint8_t[:n_bytes]> rawbytes
    compressed.free_data = True

    return (
        np.asarray(compressed),
        flat_starts,
        flat_nbytes,
    )


def wrap_encode_i32_threaded(
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] flatdata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.uint32_t level,
):
    """Wrapper around the C int32 encode function (threaded version).

    This does some data setup, but works with a flat-packed version of
    the input array.

    Args:
        flatdata (array):  The 1D reshaped view of the data.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        level (uint32_t):  The compression level (0-8).

    Returns:
        (tuple): The (compressed bytes, flat-packed starting bytes,
            flat-packed stream bytes).

    """
    # Allocate the array of starts
    cdef cnp.ndarray flat_starts = np.empty(n_stream, dtype=np.int64, order="C")
    cdef cnp.ndarray flat_nbytes = np.empty(n_stream, dtype=np.int64, order="C")

    cdef int64_t n_bytes
    cdef unsigned char * rawbytes
    cdef int errcode = 0

    errcode = encode_i32_threaded(
        <cnp.int32_t *>flatdata.data,
        n_stream,
        stream_size,
        level,
        &n_bytes,
        <cnp.int64_t *>flat_starts.data,
        &rawbytes,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Encoding failed, return code = {errcode}"
        raise RuntimeError(msg)

    # Compute the bytes per stream
    flat_nbytes[:-1] = np.diff(flat_starts)
    flat_nbytes[-1] = n_bytes - flat_starts[-1]

    # Wrap the returned C-allocated buffers so that they are properly garbage
    # collected.
    cdef cvarray compressed = <cnp.uint8_t[:n_bytes]> rawbytes
    compressed.free_data = True

    return (
        np.asarray(compressed),
        flat_starts,
        flat_nbytes,
    )


def wrap_encode_i64(
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] flatdata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.uint32_t level,
):
    """Wrapper around the C int64 encode function.

    This does some data setup, but works with a flat-packed version of
    the input array.

    Args:
        flatdata (array):  The 1D reshaped view of the data.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        level (uint32_t):  The compression level (0-8).

    Returns:
        (tuple): The (compressed bytes, flat-packed starting bytes,
            flat-packed stream bytes).

    """
    # Allocate the array of starts
    cdef cnp.ndarray flat_starts = np.empty(n_stream, dtype=np.int64, order="C")
    cdef cnp.ndarray flat_nbytes = np.empty(n_stream, dtype=np.int64, order="C")

    cdef int64_t n_bytes
    cdef unsigned char * rawbytes
    cdef int errcode = 0

    errcode = encode_i64(
        <cnp.int64_t *>flatdata.data,
        n_stream,
        stream_size,
        level,
        &n_bytes,
        <cnp.int64_t *>flat_starts.data,
        &rawbytes,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Encoding failed, return code = {errcode}"
        raise RuntimeError(msg)

    # Compute the bytes per stream
    flat_nbytes[:-1] = np.diff(flat_starts)
    flat_nbytes[-1] = n_bytes - flat_starts[-1]

    # Wrap the returned C-allocated buffers so that they are properly garbage
    # collected.
    cdef cvarray compressed = <cnp.uint8_t[:n_bytes]> rawbytes
    compressed.free_data = True

    return (
        np.asarray(compressed),
        flat_starts,
        flat_nbytes,
    )


def wrap_encode_i64_threaded(
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] flatdata,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.uint32_t level,
):
    """Wrapper around the C int64 encode function (threaded version).

    This does some data setup, but works with a flat-packed version of
    the input array.

    Args:
        flatdata (array):  The 1D reshaped view of the data.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        level (uint32_t):  The compression level (0-8).

    Returns:
        (tuple): The (compressed bytes, flat-packed starting bytes,
            flat-packed stream bytes).

    """
    # Allocate the array of starts
    cdef cnp.ndarray flat_starts = np.empty(n_stream, dtype=np.int64, order="C")
    cdef cnp.ndarray flat_nbytes = np.empty(n_stream, dtype=np.int64, order="C")

    cdef int64_t n_bytes
    cdef unsigned char * rawbytes
    cdef int errcode = 0

    errcode = encode_i64_threaded(
        <cnp.int64_t *>flatdata.data,
        n_stream,
        stream_size,
        level,
        &n_bytes,
        <cnp.int64_t *>flat_starts.data,
        &rawbytes,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Encoding failed, return code = {errcode}"
        raise RuntimeError(msg)

    # Compute the bytes per stream
    flat_nbytes[:-1] = np.diff(flat_starts)
    flat_nbytes[-1] = n_bytes - flat_starts[-1]

    # Wrap the returned C-allocated buffers so that they are properly garbage
    # collected.
    cdef cvarray compressed = <cnp.uint8_t[:n_bytes]> rawbytes
    compressed.free_data = True

    return (
        np.asarray(compressed),
        flat_starts,
        flat_nbytes,
    )


def encode_flac(data, int level, bool use_threads=False):
    """Compress an integer array to a FLAC representation.

    The input array must be C-contiguous in memory.  The last dimension is the one
    which will be compressed.  The returned array of starting bytes will have the same
    shape as the leading (uncompressed) dimensions of `data`.  The array of number of
    bytes per stream is returned as a convenience to allow easier extraction of
    subsets of streams within the larger compressed blob.

    The returned starts and nbytes arrays are always at least a 1D array, even if
    the data consists of a single stream.

    Args:
        data (numpy.ndarray):  The array of 32bit or 64bit integers.
        level (int):  The FLAC compression level (0-8).
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

    Returns:
        (tuple):  The (compressed bytestream, stream starting bytes, stream nbytes).

    """
    if data.dtype != flac_i32_dtype and data.dtype != flac_i64_dtype:
        msg = "Only 32bit or 64bit integer data is supported"
        raise RuntimeError(msg)
    if not data.data.c_contiguous:
        msg = "Only C-contiguous arrays are supported"
        raise RuntimeError(msg)
    if level < 0 or level > 8:
        msg = "FLAC only supports compression levels 0-8"
        raise RuntimeError(msg)

    stream_size = data.shape[-1]
    if len(data.shape[:-1]) == 0:
        n_stream = 1
        starts_shape = (1,)
    else:
        n_stream = np.prod(data.shape[:-1])
        starts_shape = data.shape[:-1]
    flatdata = data.reshape((-1,))

    if use_threads:
        if data.dtype == flac_i32_dtype:
            compressed, flatstarts, flatnbytes = wrap_encode_i32_threaded(
                flatdata, n_stream, stream_size, level
            )
        else:
            compressed, flatstarts, flatnbytes = wrap_encode_i64_threaded(
                flatdata, n_stream, stream_size, level
            )
    else:
        if data.dtype == flac_i32_dtype:
            compressed, flatstarts, flatnbytes = wrap_encode_i32(
                flatdata, n_stream, stream_size, level
            )
        else:
            compressed, flatstarts, flatnbytes = wrap_encode_i64(
                flatdata, n_stream, stream_size, level
            )

    # Reshape and return
    return (
        compressed,
        flatstarts.reshape(starts_shape),
        flatnbytes.reshape(starts_shape)
    )


def wrap_decode_i32(
    cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] compressed,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] starts,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] nbytes,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.int64_t first_sample,
    cnp.int64_t last_sample,
    bool use_threads,
):
    """Wrapper around the C int32 decode function.

    This works with flat-packed versions of the arrays.

    Args:
        compressed (array):  The array of compressed bytes.
        starts (array):  The array of starting bytes.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        first_sample (int64_t):  The first sample to decode.  Negative value indicates
            this parameter is unused and the whole stream should be decoded.
        last_sample (int64_t):  The last sample to decode (exclusive).  Negative value
            indicates this parameter is unused and the whole stream should be decoded.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

    Returns:
        (array):  The flat-packed int32 decompressed array.

    """
    cdef int64_t n_decode = stream_size
    if first_sample >= 0 and last_sample >= 0:
        n_decode = last_sample - first_sample

    cdef int64_t flat_size = n_stream * n_decode

    # Pre-allocate the output
    cdef cnp.ndarray output = np.empty(flat_size, dtype=flac_i32_dtype, order="C")

    cdef int errcode = 0
    errcode = decode_i32(
        <cnp.uint8_t *>compressed.data,
        <cnp.int64_t *>starts.data,
        <cnp.int64_t *>nbytes.data,
        n_stream,
        stream_size,
        first_sample,
        last_sample,
        <cnp.int32_t *>output.data,
        use_threads,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Decoding failed, return code = {errcode}"
        raise RuntimeError(msg)
    return output

def wrap_decode_i64(
    cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] compressed,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] starts,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] nbytes,
    cnp.int64_t n_stream,
    cnp.int64_t stream_size,
    cnp.int64_t first_sample,
    cnp.int64_t last_sample,
    bool use_threads,
):
    """Wrapper around the C int64 decode function.

    This works with flat-packed versions of the arrays.

    Args:
        compressed (array):  The array of compressed bytes.
        starts (array):  The array of starting bytes.
        n_stream (int64_t):  The number of streams.
        stream_size (int64_t):  The length of each stream.
        first_sample (int64_t):  The first sample to decode.  Negative value indicates
            this parameter is unused and the whole stream should be decoded.
        last_sample (int64_t):  The last sample to decode (exclusive).  Negative value
            indicates this parameter is unused and the whole stream should be decoded.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.

    Returns:
        (array):  The flat-packed int64 decompressed array.

    """
    cdef int64_t n_decode = stream_size
    if first_sample >= 0 and last_sample >= 0:
        n_decode = last_sample - first_sample

    cdef int64_t flat_size = n_stream * n_decode

    # Pre-allocate the output
    cdef cnp.ndarray output = np.empty(flat_size, dtype=flac_i64_dtype, order="C")

    cdef int errcode = 0
    errcode = decode_i64(
        <cnp.uint8_t *>compressed.data,
        <cnp.int64_t *>starts.data,
        <cnp.int64_t *>nbytes.data,
        n_stream,
        stream_size,
        first_sample,
        last_sample,
        <cnp.int64_t *>output.data,
        use_threads,
    )

    if errcode != 0:
        # FIXME: change error codes so we can print a message here
        msg = f"Decoding failed, return code = {errcode}"
        raise RuntimeError(msg)
    return output

def decode_flac(
    compressed,
    starts,
    nbytes,
    int stream_size,
    int first_sample=-1,
    int last_sample=-1,
    bool use_threads=False,
    bool is_int64=False,
):
    """Decompress a FLAC compressed bytestream.

    The shape of the input `starts` is used to determine the leading dimensions of
    the output array.  The `stream_size` is the decompressed size of the final
    dimension.

    Even if there is only one stream, the starts and nbytes arrays should be 1D.

    Args:
        compressed (numpy.ndarray):  The array of compressed bytes.
        starts (numpy.ndarray):  The array of starting bytes in the bytestream.
        nbytes (numpy.ndarray):  The array of number of bytes in the bytestream.
        stream_size (int):  The length of the decompressed final dimension.
        first_sample (int):  The first sample to decode along the final dimension.
            Negative value indicates this parameter is unused and the whole stream
            should be decoded.
        last_sample (int):  The last sample to decode along the final dimension
            (exclusive).  Negative value indicates this parameter is unused and the
            whole stream should be decoded.
        use_threads (bool):  If True, use OpenMP threads to parallelize decoding.
            This is only beneficial for large arrays.
        is_int64 (bool):  If True, the compressed stream contains 64bit integers
            encoded as 2 channels.

    Returns:
        (array):  The decompressed array of int32 or int64 data.

    """
    if compressed.dtype != compressed_dtype:
        msg = "Compressed data should be of type uint8"
        raise RuntimeError(msg)
    if not compressed.data.c_contiguous:
        msg = "Only C-contiguous arrays are supported"
        raise RuntimeError(msg)
    if starts.dtype != offset_dtype:
        msg = "starts data should be of type int64"
        raise RuntimeError(msg)
    if not starts.data.c_contiguous:
        msg = "Only C-contiguous arrays are supported"
        raise RuntimeError(msg)
    if nbytes.dtype != offset_dtype:
        msg = "nbytes data should be of type int64"
        raise RuntimeError(msg)
    if not nbytes.data.c_contiguous:
        msg = "Only C-contiguous arrays are supported"
        raise RuntimeError(msg)
    if stream_size <= 0:
        msg = "You must specify the non-zero output stream size"
        raise RuntimeError(msg)

    if len(compressed.shape) != 1:
        msg = "Compressed byte array should be one dimensional"
        raise RuntimeError(msg)

    n_decode = stream_size
    if first_sample >= 0 and last_sample >= 0:
        # We are decoding a slice of samples.
        if last_sample > stream_size:
            msg = "last_sample is beyond end of stream"
            raise RuntimeError(msg)
        if first_sample > stream_size - 1:
            msg = "first_sample is beyond last element of stream"
            raise RuntimeError(msg)
        if first_sample >= last_sample:
            msg = "first_sample is larger than last_sample"
            raise RuntimeError(msg)
        n_decode = last_sample - first_sample

    output_shape = starts.shape + (n_decode,)
    cdef int64_t cstream_size = stream_size
    cdef int64_t cfirst_sample = first_sample
    cdef int64_t clast_sample = last_sample
    cdef int64_t n_stream = np.prod(starts.shape)
    flat_starts = starts.reshape((-1,))
    flat_nbytes = nbytes.reshape((-1,))

    if is_int64:
        flat_output = wrap_decode_i64(
            compressed,
            flat_starts,
            flat_nbytes,
            n_stream,
            cstream_size,
            cfirst_sample,
            clast_sample,
            use_threads,
        )
    else:
        flat_output = wrap_decode_i32(
            compressed,
            flat_starts,
            flat_nbytes,
            n_stream,
            cstream_size,
            cfirst_sample,
            clast_sample,
            use_threads,
        )

    # Reshape and return
    return flat_output.reshape(output_shape)


