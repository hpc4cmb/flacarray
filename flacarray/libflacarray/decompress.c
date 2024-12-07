// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <stdbool.h>

#include <FLAC/stream_decoder.h>

#include "flacarray.h"


// This structure is used as the client data for BOTH the read and write
// callback functions below.

typedef struct {
    unsigned char const * input;
    int64_t n_stream;
    int64_t n_decode;
    int64_t cur_stream;
    int64_t stream_start;
    int64_t stream_end;
    int64_t stream_pos;
    int64_t decomp_nelem;
    int32_t * decompressed;
    int32_t err;
} dec_callback_data;


// Read callback function, called by the decoder for each chunk to request
// more bytes from the bytestream.
FLAC__StreamDecoderReadStatus dec_read_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__byte buffer[],
    uint64_t * bytes,
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;

    unsigned char const * input = callback_data->input;
    int64_t pos = callback_data->stream_pos;
    int64_t remaining = callback_data->stream_end - pos;

    // The bytes requested by the decoder
    int64_t n_buffer = (*bytes);

    if (remaining == 0) {
        // No data left
        (*bytes) = 0;
        return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
    } else {
        // We have some data left
        if (n_buffer == 0) {
            // ... but there is no place to put it!
            callback_data->err = ERROR_DECODE_READ_ZEROBUF;
            return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
        } else {
            if (remaining > n_buffer) {
                memcpy(
                    (void*)buffer,
                    (void*)(input + pos),
                    n_buffer
                );
                callback_data->stream_pos += n_buffer;
                return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
            } else {
                memcpy(
                    (void*)buffer,
                    (void*)(input + pos),
                    remaining
                );
                callback_data->stream_pos += remaining;
                (*bytes) = remaining;
                return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
            }
        }
    }

    // Should never get here...
    return FLAC__STREAM_DECODER_READ_STATUS_ABORT;
}


// Write callback function, called by the decoder to process the output
// decompressed integers.
FLAC__StreamDecoderWriteStatus dec_write_callback(
    const FLAC__StreamDecoder * decoder,
    const FLAC__Frame * frame,
    const FLAC__int32 * const buffer[],
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    int64_t nelem = callback_data->decomp_nelem;
    int64_t n_decode = callback_data->n_decode;
    int32_t * decomp = callback_data->decompressed;
    uint32_t blocksize = frame->header.blocksize;

    // The number of bytes to copy might be smaller than blocksize, if we are on the
    // last block.
    int64_t n_copy = blocksize;
    if (nelem + blocksize > n_decode) {
        n_copy = n_decode - nelem;
    }

    memcpy(
        (void*)(decomp + nelem),
        (void*)buffer[0],
        n_copy * sizeof(int32_t)
    );

    // Increment our number of decoded elements
    callback_data->decomp_nelem += n_copy;

    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}


void dec_err_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__StreamDecoderErrorStatus status,
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    int64_t cur = callback_data->cur_stream;
    int64_t n_elem = callback_data->decomp_nelem;
    fprintf(
        stderr,
        "FLAC decode error (%d) on stream %ld at bytes %ld - %ld, output size %ld\n",
        status,
        cur,
        callback_data->stream_pos,
        callback_data->stream_end,
        n_elem
    );
    return;
}


FLAC__StreamDecoderSeekStatus dec_seek_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__uint64 absolute_byte_offset,
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    // Range of bytes being decoded for this stream within the full, multi-stream
    // buffer.
    int64_t stream_start = callback_data->stream_start;
    int64_t stream_end = callback_data->stream_end;
    // Convert the requested stream offset into absolute position.
    int64_t abs_request = absolute_byte_offset + stream_start;

    if (abs_request > stream_end) {
        // Position beyond the end of the stream
        return FLAC__STREAM_DECODER_SEEK_STATUS_ERROR;
    }
    // Update byte position
    callback_data->stream_pos = abs_request;
    return FLAC__STREAM_DECODER_SEEK_STATUS_OK;
}


FLAC__StreamDecoderTellStatus dec_tell_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__uint64 * absolute_byte_offset,
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    // Compute the relative position in the current stream
    int64_t rel_pos = callback_data->stream_pos - callback_data->stream_start;
    (*absolute_byte_offset) = rel_pos;
    return FLAC__STREAM_DECODER_TELL_STATUS_OK;
}


FLAC__StreamDecoderLengthStatus dec_length_callback(
    const FLAC__StreamDecoder * decoder,
    FLAC__uint64 * stream_length,
    void * client_data
) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    (*stream_length) = callback_data->stream_end - callback_data->stream_start;
    return FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
}


FLAC__bool dec_eof_callback(const FLAC__StreamDecoder * decoder, void * client_data) {
    dec_callback_data * callback_data = (dec_callback_data *)client_data;
    if (callback_data->stream_pos >= callback_data->stream_end) {
        return true;
    } else {
        return false;
    }
}


// Main decode function.  The input bytes buffer is passed along with the start byte
// and number of bytes in this buffer for each of the streams.  For performance
// reasons, the calling code should pass in the pre-allocated output data buffer to
// populate.  This allows each stream to be decompressed by separate threads and copied
// into the shared buffer. In practice, the size of this final, uncompressed dimension
// should always be available in the python classes or in the on-disk data formats.
// If decoding a slice of samples, the first_sample and last_sample should define a
// range that is less than or equal to 0...stream_size.  The last_sample parameter is
// "exclusive", similar to python slice notation.  If any errors occur, the processing
// stops, an attempt is made to free any buffers that were allocated, and an error code
// is returned which is a bitwise OR of the errors on all threads.

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
) {
    // Verify the requested sample range.
    int64_t first_decode = 0;
    int64_t n_decode = stream_size;
    if ((first_sample >= 0) && (last_sample >= 0)) {
        // We are decoding a slice of samples.
        if (last_sample > stream_size) {
            return ERROR_DECODE_SAMPLE_RANGE;
        }
        if (first_sample > stream_size - 1) {
            return ERROR_DECODE_SAMPLE_RANGE;
        }
        if (first_sample >= last_sample) {
            return ERROR_DECODE_SAMPLE_RANGE;
        }
        first_decode = first_sample;
        n_decode = last_sample - first_sample;
    }

    // This tracks the failures across all threads.
    int errors = ERROR_NONE;

    #pragma omp parallel reduction(|:errors) if(use_threads)
    {
        // Thread-local decoder
        FLAC__StreamDecoder * decoder;
        bool success;
        FLAC__StreamDecoderInitStatus status;

        // Thread-local decode buffers
        dec_callback_data callback_data;
        callback_data.input = bytes;
        callback_data.n_stream = n_stream;
        callback_data.n_decode = n_decode;
        callback_data.err = ERROR_NONE;

        #pragma omp for schedule(static)
        for (int64_t istream = 0; istream < n_stream; ++istream) {
            if (errors != ERROR_NONE) {
                // We already had a failure, skip over remaining loop iterations
                continue;
            }
            callback_data.cur_stream = istream;
            callback_data.stream_start = starts[istream];
            callback_data.stream_end = starts[istream] + nbytes[istream];
            callback_data.stream_pos = starts[istream];
            callback_data.decomp_nelem = 0;
            // Set the output buffer to the address of the beginning of this stream.
            callback_data.decompressed = data + n_decode * istream;

            decoder = FLAC__stream_decoder_new();

            status = FLAC__stream_decoder_init_stream(
                decoder,
                dec_read_callback,
                dec_seek_callback,
                dec_tell_callback,
                dec_length_callback,
                dec_eof_callback,
                dec_write_callback,
                NULL,
                dec_err_callback,
                (void *)&callback_data
            );
            if (status != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
                errors |= ERROR_DECODE_INIT;
                continue;
            }
            if (n_decode == stream_size) {
                // We are decoding all samples
                success = FLAC__stream_decoder_process_until_end_of_stream(decoder);
                if (!success) {
                    errors |= ERROR_DECODE_PROCESS;
                    continue;
                }
            } else {
                // We are decoding a slice of samples.  Seek to the start.
                success = FLAC__stream_decoder_seek_absolute(decoder, first_decode);
                if (!success) {
                    errors |= ERROR_DECODE_PROCESS;
                    continue;
                }
                // Process single frames until we have accumulated at least the desired
                // number of output samples.
                while (
                    callback_data.decomp_nelem < n_decode
                ) {
                    success = FLAC__stream_decoder_process_single(decoder);
                    if (!success) {
                        errors |= ERROR_DECODE_PROCESS;
                        continue;
                    }
                }
            }
            success = FLAC__stream_decoder_finish(decoder);
            if (!success) {
                errors |= ERROR_DECODE_FINISH;
                continue;
            }
            FLAC__stream_decoder_delete(decoder);

            // Merge any errors from the decoder callback
            errors |= callback_data.err;
        }
    }

    return errors;
}

