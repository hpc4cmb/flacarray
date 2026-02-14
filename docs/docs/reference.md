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

## Compiled Library Interfaces

The lowest-level functions in the compiled C code are exposed in the `flacarray.h` header file and the `libflacarray` library.  These are installed along with the python package, and can be linked into compiled software products.

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
