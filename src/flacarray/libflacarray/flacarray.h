// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#ifdef _OPENMP
# include <omp.h>
#endif // ifdef _OPENMP


// Error codes

#define ERROR_NONE 0
#define ERROR_ALLOC 1
#define ERROR_INVALID_LEVEL 2
#define ERROR_ZERO_NSTREAM 4
#define ERROR_ZERO_STREAMSIZE 8
#define ERROR_ENCODE_SET_COMP_LEVEL 16
#define ERROR_ENCODE_SET_BLOCK_SIZE 32
#define ERROR_ENCODE_SET_CHANNELS 64
#define ERROR_ENCODE_SET_BPS 128
#define ERROR_ENCODE_INIT 256
#define ERROR_ENCODE_PROCESS 512
#define ERROR_ENCODE_FINISH 1024
#define ERROR_ENCODE_COLLECT 2048
#define ERROR_DECODE_READ_ZEROBUF 4096
#define ERROR_DECODE_INIT 8192
#define ERROR_DECODE_PROCESS 16384
#define ERROR_DECODE_FINISH 32768
#define ERROR_DECODE_STREAMSIZE 65536
#define ERROR_DECODE_SAMPLE_RANGE 131072
#define ERROR_CONVERT_TYPE 262144

// C-language arrays with a few STL-like features.

typedef struct {
    int64_t size;
    int64_t n_elem;
    unsigned char * data;
} ArrayUint8;

typedef struct {
    int64_t size;
    int64_t n_elem;
    int32_t * data;
} ArrayInt32;

ArrayUint8 * create_array_uint8(int64_t start_size);
void destroy_array_uint8(ArrayUint8 * obj);
int resize_array_uint8(ArrayUint8 * obj, int64_t new_size);

ArrayInt32 * create_array_int32(int64_t start_size);
void destroy_array_int32(ArrayInt32 * obj);
int resize_array_int32(ArrayInt32 * obj, int64_t new_size);

// Encoding

int encode(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

int encode_threaded(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

// Decoding

int decode(
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

// Type conversion

int int64_to_int32(
    int64_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    int32_t * output,
    int64_t * offsets
);

int float32_to_int32(
    float const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * quanta,
    int32_t * output,
    float * offsets,
    float * gains
);

int float64_to_int32(
    double const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * quanta,
    int32_t * output,
    double * offsets,
    double * gains
);

void int32_to_int64(
    int32_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    int64_t const * offsets,
    int64_t * output
);

void int32_to_float64(
    int32_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * offsets,
    double const * gains,
    double * output
);

void int32_to_float32(
    int32_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * offsets,
    float const * gains,
    float * output
);

