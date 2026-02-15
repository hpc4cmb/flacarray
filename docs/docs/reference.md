# API Reference

The `flacarray` package consists of a primary class (`FlacArray`) plus a
variety of helper functions.

## Compressed Array Representation

The `FlacArray` class stores a compressed representation of an N dimensional
array where the last dimension consists of "streams" of numbers to be
compressed.

::: flacarray.FlacArray

## Direct I/O

Sometimes code has no need to store compressed arrays in memory. Instead, it
may be desirable to have full arrays in memory and compressed arrays on disk.
In those situations, you can use several helper functions to write and read
numpy arrays directly to / from files.

### HDF5

You can write to / read from an h5py Group using functions in the `hdf5`
submodule.

::: flacarray.hdf5.write_array

::: flacarray.hdf5.read_array

### Zarr

You can write to / read from a zarr hierarch Group using functions in the
`zarr` submodule.

::: flacarray.zarr.write_array

::: flacarray.zarr.read_array

## Interactive Tools

The `flacarray.demo` submodule contains a few helper functions that are not
imported by default. You will need to have optional dependencies (matplotlib)
installed to use the visualization tools. For testing, it is convenient to
generate arrays consisting of random timestreams with some structure. The
`create_fake_data` function can be used for this.

::: flacarray.demo.create_fake_data

Most data arrays in practice have 2 or 3 dimensions. If the number of streams
is relatively small, then an uncompressed array can be plotted with the
`plot_data` function.

::: flacarray.demo.plot_data

## Low-Level Tools

For specialized use cases, you can also work directly with the compressed
bytestream and auxiliary arrays and convert to / from numpy arrays.

::: flacarray.compress.array_compress

::: flacarray.decompress.array_decompress

::: flacarray.decompress.array_decompress_slice

## Notes About Threading

The optional threading in flacarray uses OpenMP to distribute *entire streams*
among threads. OpenMP was chosen since some downstream codes using `flacarray`
are hybrid MPI / OpenMP codebases and it is convenient to set the threading
concurrency for all software in a running job. However, the encode / decode
operations are very fast, and for small arrays the threading overhead can
actually slow things down. For large arrays (hundreds of streams, each with
millions of samples), using threads does provide speed ups (especially for
encode operations). For these reasons, the threading options for all functions
in the API are disabled by default.

Newer versions of libFLAC (>=1.5.0) include support for threaded encodes. This
threading is done over frames within a single stream. Currently flacarray does
not use this, since it might introduce oversubscription of cores if both
threading models were enabled. Future flacarray developments might make use of
libFLAC threads if they show real-world performance improvements.

## Compiled Library Interfaces

The lowest-level functions in the compiled C code are exposed in the
`flacarray.h` header file and the `libflacarray` library. These are installed
along with the python package, and can be linked into compiled software
products. For an example of using these functions, see the compiled test case in
`src/flacarray/libflacarray/test.c`

### Encoding

The encoding functions have very different internals for the unthreaded and
threaded cases. For this reason, they are split into different functions. The
32bit functions encode data to a single FLAC channel, while the 64bit functions
use two interleaved FLAC channels.

``` c
/*
 * Encode multiple 32bit integer streams.
 *
 * The input streams are passed as a single (concatenated), flat-packed buffer.
 * This input `data` should have length `n_stream` X `stream_size`.
 *
 * The `n_bytes` and `starts` should be pre-allocated to hold one value per
 * stream.  These will be populated by the function.  The `bytes` pointer
 * is the address of a pointer that will be malloc'd and populated with
 * the compressed bytes.  The `starts` values are the starting byte offset
 * of each stream in this output buffer.  The `n_bytes` are the number
 * of bytes following the starting point for each stream.
 *
 * @param[in] data The input integer data.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param level The FLAC compression level.
 * @param[in] n_bytes The number of bytes per stream (filled by this function).
 * @param[in] starts The starting byte per stream (filled by this function).
 * @param[out] bytes The array of pointers to the bytes for each stream.
 */
int encode_i32(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);
```

Threaded version:

``` c
/*
 * Encode multiple 32bit integer streams with threads.
 *
 * The streams are distributed to the available threads, which independently
 * accumulate the compressed bytes for their assigned streams.  These
 * thread-local compressed streams are then copied into the final output.
 *
 * The input streams are passed as a single (concatenated), flat-packed buffer.
 * This input `data` should have length `n_stream` X `stream_size`.
 *
 * The `n_bytes` and `starts` should be pre-allocated to hold one value per
 * stream.  These will be populated by the function.  The `bytes` pointer
 * is the address of a pointer that will be malloc'd and populated with
 * the compressed bytes.  The `starts` values are the starting byte offset
 * of each stream in this output buffer.  The `n_bytes` are the number
 * of bytes following the starting point for each stream.
 *
 * @param[in] data The input integer data.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param level The FLAC compression level.
 * @param[in] n_bytes The number of bytes per stream (filled by this function).
 * @param[in] starts The starting byte per stream (filled by this function).
 * @param[out] bytes The array of pointers to the bytes for each stream.
 */
int encode_i32_threaded(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);
```

The 64bit encode functions are the same except for the input data type:

``` c
/*
 * Encode multiple 64bit integer streams.
 *
 * The input streams are passed as a single (concatenated), flat-packed buffer.
 * This input `data` should have length `n_stream` X `stream_size`.
 *
 * The `n_bytes` and `starts` should be pre-allocated to hold one value per
 * stream.  These will be populated by the function.  The `bytes` pointer
 * is the address of a pointer that will be malloc'd and populated with
 * the compressed bytes.  The `starts` values are the starting byte offset
 * of each stream in this output buffer.  The `n_bytes` are the number
 * of bytes following the starting point for each stream.
 *
 * @param[in] data The input integer data.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param level The FLAC compression level.
 * @param[in] n_bytes The number of bytes per stream (filled by this function).
 * @param[in] starts The starting byte per stream (filled by this function).
 * @param[out] bytes The array of pointers to the bytes for each stream.
 */
int encode_i64(
    int64_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);
```

``` c
/*
 * Encode multiple 64bit integer streams with threads.
 *
 * The streams are distributed to the available threads, which independently
 * accumulate the compressed bytes for their assigned streams.  These
 * thread-local compressed streams are then copied into the final output.
 *
 * The input streams are passed as a single (concatenated), flat-packed buffer.
 * This input `data` should have length `n_stream` X `stream_size`.
 *
 * The `n_bytes` and `starts` should be pre-allocated to hold one value per
 * stream.  These will be populated by the function.  The `bytes` pointer
 * is the address of a pointer that will be malloc'd and populated with
 * the compressed bytes.  The `starts` values are the starting byte offset
 * of each stream in this output buffer.  The `n_bytes` are the number
 * of bytes following the starting point for each stream.
 *
 * @param[in] data The input integer data.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param level The FLAC compression level.
 * @param[in] n_bytes The number of bytes per stream (filled by this function).
 * @param[in] starts The starting byte per stream (filled by this function).
 * @param[out] bytes The array of pointers to the bytes for each stream.
 */
int encode_i64_threaded(
    int64_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);
```

### Decoding

The decoding functions use a boolean switch to select whether to use OpenMP
threads. This is possible since the internal code is nearly the same in both the
threaded and non-threaded cases. The output buffer size is known and each thread
can independently work on decoding streams and populating slices of the output.

``` c
/*
 * Decode multiple 32bit integer streams.
 *
 * The input `bytes` are the concatenated buffers for all streams.
 *
 * The `starts` values are the starting byte offset of each stream in `bytes`.
 * The `n_bytes` are the number of bytes following the starting point for each
 * stream.
 *
 * The `first_sample` and `last_sample` arguments control the subset of values
 * that are extracted from each stream.  To extract all values, set `first_sample`
 * to zero and `last_sample` to `stream_size`.
 *
 * The output data buffer should be pre-allocated to hold
 * `n_stream` X (`last_sample` - `first_sample`) elements.
 *
 * @param[in] bytes The concatenated compressed bytes for all streams.
 * @param[in] n_bytes The number of bytes per stream.
 * @param[in] starts The starting byte per stream.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param first_sample The first output sample to decode for all streams.
 * @param last_sample The last output sample (exclusive) to decode for all streams.
 * @param[out] data The output integer data.
 * @param[in] use_threads If true, used threads for decoding.
 */
int decode_i32(
    unsigned char * const bytes,
    int64_t * const starts,
    int64_t * const nbytes,
    int64_t n_stream,
    int64_t stream_size,
    int64_t first_sample,
    int64_t last_sample,
    int32_t * data,
    bool use_threads
);
```

``` c
/*
 * Decode multiple 64bit integer streams.
 *
 * The input `bytes` are the concatenated buffers for all streams.
 *
 * The `starts` values are the starting byte offset of each stream in `bytes`.
 * The `n_bytes` are the number of bytes following the starting point for each
 * stream.
 *
 * The `first_sample` and `last_sample` arguments control the subset of values
 * that are extracted from each stream.  To extract all values, set `first_sample`
 * to zero and `last_sample` to `stream_size`.
 *
 * The output data buffer should be pre-allocated to hold
 * `n_stream` X (`last_sample` - `first_sample`) elements.
 *
 * @param[in] bytes The concatenated compressed bytes for all streams.
 * @param[in] n_bytes The number of bytes per stream.
 * @param[in] starts The starting byte per stream.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param first_sample The first output sample to decode for all streams.
 * @param last_sample The last output sample (exclusive) to decode for all streams.
 * @param[out] data The output integer data.
 * @param[in] use_threads If true, used threads for decoding.
 */
int decode_i64(
    unsigned char * const bytes,
    int64_t * const starts,
    int64_t * const nbytes,
    int64_t n_stream,
    int64_t stream_size,
    int64_t first_sample,
    int64_t last_sample,
    int64_t * data,
    bool use_threads
);
```

### Type Conversions

Flacarray supports encoding floating point data as compressed integers. This is
done by setting the `quanta` for all streams. This number represents the
smallest floating point value that can be preserved in a lossless way by
encoding to integers and back. The compiled library has helper functions that
can be used to convert back and forth. 32bit floating point values are converted
to 32bit integers and 64bit floating point values are converted to 64bit
integers.

``` c
/*
 * Convert 32bit floating point data to integers.
 *
 * The `input` data is the concatenated buffers for all streams.  The `quanta`
 * array is the smallest floating point value for each stream that will be
 * preserved in a lossless way going from float to int and back.
 *
 * The `output` data buffer should be pre-allocated to hold
 * `n_stream` X `stream_size` elements.  The `offsets` and `gains` should be
 * pre-allocated to hold one value per stream.
 *
 * @param[in] input The concatenated input data from multiple streams.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param[in] quanta The quantization value for each stream.
 * @param[out] output The output integer data.
 * @param[out] offsets The floating point offset subtracted from each input stream.
 * @param[out] gains The gain applied to each input stream before truncation.
 */
int float32_to_int32(
    float const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * quanta,
    int32_t * output,
    float * offsets,
    float * gains
);
```

``` c
/*
 * Convert 64bit floating point data to integers.
 *
 * The `input` data is the concatenated buffers for all streams.  The `quanta`
 * array is the smallest floating point value for each stream that will be
 * preserved in a lossless way going from float to int and back.
 *
 * The `output` data buffer should be pre-allocated to hold
 * `n_stream` X `stream_size` elements.  The `offsets` and `gains` should be
 * pre-allocated to hold one value per stream.
 *
 * @param[in] input The concatenated input data from multiple streams.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param[in] quanta The quantization value for each stream.
 * @param[out] output The output integer data.
 * @param[out] offsets The floating point offset subtracted from each input stream.
 * @param[out] gains The gain applied to each input stream before truncation.
 */
int float64_to_int64(
    double const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * quanta,
    int64_t * output,
    double * offsets,
    double * gains
);
```

``` c
/*
 * Convert 32bit integers to floating point data.
 *
 * The `input` data is the concatenated buffers for all streams.  To restore
 * the original floating point data, the integer values are scaled by 1/gain
 * and then the offset is added.
 *
 * The `output` data buffer should be pre-allocated to hold
 * `n_stream` X `stream_size` elements.
 *
 * @param[in] input The concatenated input data from multiple streams.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param[in] offsets The floating point offset subtracted from each original stream.
 * @param[in] gains The gain applied to each original stream before truncation.
 * @param[out] output The restored floating point data.
 */
void int32_to_float32(
    int32_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * offsets,
    float const * gains,
    float * output
);
```

``` c
/*
 * Convert 64bit integers to floating point data.
 *
 * The `input` data is the concatenated buffers for all streams.  To restore
 * the original floating point data, the integer values are scaled by 1/gain
 * and then the offset is added.
 *
 * The `output` data buffer should be pre-allocated to hold
 * `n_stream` X `stream_size` elements.
 *
 * @param[in] input The concatenated input data from multiple streams.
 * @param n_stream The number of input streams.
 * @param stream_size The length of each stream.
 * @param[in] offsets The floating point offset subtracted from each original stream.
 * @param[in] gains The gain applied to each original stream before truncation.
 * @param[out] output The restored floating point data.
 */
void int64_to_float64(
    int64_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * offsets,
    double const * gains,
    double * output
);
```

### Error Codes

The error codes returned by the compiled functions are a bitwise OR of multiple possible error conditions.  These are defined in terms of the bit location in the integer error code values:

``` c
// Success (No error).
#define ERROR_NONE 0

// Memory allocation error.
#define ERROR_ALLOC (1 << 0)

// Invalid FLAC compression level.
#define ERROR_INVALID_LEVEL (1 << 1)

// Number of streams is zero.
#define ERROR_ZERO_NSTREAM (1 << 2)

// Streamsize is zero.
#define ERROR_ZERO_STREAMSIZE (1 << 3)

// Unable to set encoder compression level.
#define ERROR_ENCODE_SET_COMP_LEVEL (1 << 4)

// Unable to set encoder block size.
#define ERROR_ENCODE_SET_BLOCK_SIZE (1 << 5)

// Unable to set number of encoder channels.
#define ERROR_ENCODE_SET_CHANNELS (1 << 6)

// Unable to set encoder bits per sample.
#define ERROR_ENCODE_SET_BPS (1 << 7)

// Unable to initialize encoder.
#define ERROR_ENCODE_INIT (1 << 8)

// Failed to run encoder.
#define ERROR_ENCODE_PROCESS (1 << 9)

// Failed to finish encoding
#define ERROR_ENCODE_FINISH (1 << 10)

// Failed to collect thread-local results.
#define ERROR_ENCODE_COLLECT (1 << 11)

// Decoder failed to request bytes for remaining data.
#define ERROR_DECODE_READ_ZEROBUF (1 << 12)

// Failed to initial decoder.
#define ERROR_DECODE_INIT (1 << 13)

// Failed to process decoder bytes.
#define ERROR_DECODE_PROCESS (1 << 14)

// Failed to finish decoding.
#define ERROR_DECODE_FINISH (1 << 15)

// Decode stream size is invalid.
#define ERROR_DECODE_STREAMSIZE (1 << 16)

// Decode sample range is invalid.
#define ERROR_DECODE_SAMPLE_RANGE (1 << 17)

// Unable to seek in bytestream.
#define ERROR_DECODE_SEEK (1 << 18)

// Unable to convert data to desired type.
#define ERROR_CONVERT_TYPE (1 << 19)
```
