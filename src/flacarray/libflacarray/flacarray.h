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

#include <FLAC/stream_encoder.h>
#include <FLAC/stream_decoder.h>


// Error codes

#define ERROR_NONE 0
#define ERROR_ALLOC (1 << 0)
#define ERROR_INVALID_LEVEL (1 << 1)
#define ERROR_ZERO_NSTREAM (1 << 2)
#define ERROR_ZERO_STREAMSIZE (1 << 3)
#define ERROR_ENCODE_SET_COMP_LEVEL (1 << 4)
#define ERROR_ENCODE_SET_BLOCK_SIZE (1 << 5)
#define ERROR_ENCODE_SET_CHANNELS (1 << 6)
#define ERROR_ENCODE_SET_BPS (1 << 7)
#define ERROR_ENCODE_INIT (1 << 8)
#define ERROR_ENCODE_PROCESS (1 << 9)
#define ERROR_ENCODE_FINISH (1 << 10)
#define ERROR_ENCODE_COLLECT (1 << 11)
#define ERROR_DECODE_READ_ZEROBUF (1 << 12)
#define ERROR_DECODE_INIT (1 << 13)
#define ERROR_DECODE_PROCESS (1 << 14)
#define ERROR_DECODE_FINISH (1 << 15)
#define ERROR_DECODE_STREAMSIZE (1 << 16)
#define ERROR_DECODE_SAMPLE_RANGE (1 << 17)
#define ERROR_DECODE_SEEK (1 << 18)
#define ERROR_CONVERT_TYPE (1 << 19)

// C-language arrays with a few STL-like features.

typedef struct {
    int64_t size;
    int64_t n_elem;
    unsigned char * data;
} ArrayUint8;

ArrayUint8 * create_array_uint8(int64_t start_size);
void destroy_array_uint8(ArrayUint8 * obj);
int resize_array_uint8(ArrayUint8 * obj, int64_t new_size);

// Encoding

// Callback structure to store the output encoded bytes.
typedef struct {
    int64_t last_stream;
    int64_t cur_stream;
    int64_t * stream_offsets;
    ArrayUint8 * compressed;
} enc_callback_data;

typedef struct {
    int64_t n_stream;
    int64_t cur_stream;
    ArrayUint8 ** compressed;
} enc_threaded_callback_data;

FLAC__StreamEncoderWriteStatus enc_write_callback(
    const FLAC__StreamEncoder * encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    uint32_t samples,
    uint32_t current_frame,
    void * client_data
);

FLAC__StreamEncoderWriteStatus enc_threaded_write_callback(
    const FLAC__StreamEncoder * encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    uint32_t samples,
    uint32_t current_frame,
    void * client_data
);

void free_compressed_buffers(ArrayUint8 ** buffers, int64_t n_stream);

int encode(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t n_channels,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

int encode_threaded(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t n_channels,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

// Decoding

// This structure is used as the client data for BOTH the read and write
// callback functions.

typedef struct {
    unsigned char const * input;
    // Total number of streams
    int64_t n_stream;
    // The number of samples to decode from all streams
    int64_t n_decode;
    // The number of channels in a single sample of a single stream
    uint32_t n_channels;
    // The current stream that is being decoded
    int64_t cur_stream;
    // The starting byte of the current stream in the compressed input
    int64_t stream_start;
    // The ending byte of the current stream in the compressed input
    int64_t stream_end;
    // The current byte position in the compressed input
    int64_t stream_pos;
    // The number of decompressed samples processed so far in this stream
    int64_t decomp_nelem;
    // The decompressed and interleaved output for the current stream.  This
    // points to the beginning of the output stream in the larger output
    // buffer, and each stream has n_decode * n_channels int32 values.
    int32_t * decompressed;
    // The current error state
    int32_t err;
} dec_callback_data;

FLAC__StreamDecoderReadStatus dec_read_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__byte buffer[],
    size_t * bytes,
    void * client_data
);

FLAC__StreamDecoderWriteStatus dec_write_callback(
    const FLAC__StreamDecoder * decoder,
    const FLAC__Frame * frame,
    const FLAC__int32 * const buffer[],
    void * client_data
);

void dec_err_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__StreamDecoderErrorStatus status,
    void * client_data
);

FLAC__StreamDecoderSeekStatus dec_seek_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__uint64 absolute_byte_offset,
    void * client_data
);

FLAC__StreamDecoderTellStatus dec_tell_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__uint64 * absolute_byte_offset,
    void * client_data
);

FLAC__StreamDecoderLengthStatus dec_length_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__uint64 * stream_length,
    void * client_data
);

FLAC__bool dec_eof_callback(const FLAC__StreamDecoder * decoder, void * client_data);

int decode(
    unsigned char * const bytes,
    int64_t * const starts,
    int64_t * const nbytes,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t n_channels,
    int64_t first_sample,
    int64_t last_sample,
    int32_t * data,
    bool use_threads
);

// Helper wrappers for int32 and int64 encode / decode.  int64 data is encoded
// as 2 interleaved channels.

bool is_little_endian();

int get_interleaved(int64_t n_elem, int64_t const * data, int32_t ** interleaved);

void copy_interleaved_64_to_32(int64_t n_elem, int64_t * input, int32_t * output);

void copy_interleaved_32_to_64(int64_t n_elem, int32_t * input, int64_t * output);

void free_interleaved(int32_t * interleaved);

int encode_i32(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

int encode_i32_threaded(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

int encode_i64(
    int64_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

int encode_i64_threaded(
    int64_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
);

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

// Type conversion

int float32_to_int32(
    float const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * quanta,
    int32_t * output,
    float * offsets,
    float * gains
);

int float64_to_int64(
    double const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * quanta,
    int64_t * output,
    double * offsets,
    double * gains
);

void int64_to_float64(
    int64_t const * input,
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

