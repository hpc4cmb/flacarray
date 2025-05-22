// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <flacarray.h>


// Verify function.

int verify(
    unsigned char * const bytes,
    int64_t * const starts,
    int64_t * const nbytes,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t n_channels,
    int64_t first_sample,
    int64_t last_sample
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

    // Decoder
    FLAC__StreamDecoder * decoder;
    bool success;
    FLAC__StreamDecoderInitStatus status;

    // Decode buffers
    dec_callback_data callback_data;
    callback_data.input = bytes;
    callback_data.n_stream = n_stream;
    callback_data.n_decode = n_decode;
    callback_data.n_channels = n_channels;
    callback_data.err = ERROR_NONE;

    // Allocate a temporary data buffer
    int32_t * decompressed = (int32_t *)malloc(
        n_stream * n_decode * n_channels * sizeof(int32_t));
    if (decompressed == NULL) {
        fprintf(stderr, "Failed to allocate temp data for verification\n");
    }

    for (int64_t istream = 0; istream < n_stream; ++istream) {
        if (errors != ERROR_NONE) {
            // We already had a failure, skip over remaining loop iterations
            continue;
        }
        fprintf(stderr, "Verifying stream %ld:\n", istream);
        fprintf(stderr, "  start byte = %ld\n", starts[istream]);
        fprintf(stderr, "  end byte = %ld\n", starts[istream] + nbytes[istream]);
        fprintf(
            stderr, "  output start element = %ld\n", istream * n_decode * n_channels
        );
        callback_data.cur_stream = istream;
        callback_data.stream_start = starts[istream];
        callback_data.stream_end = starts[istream] + nbytes[istream];
        callback_data.stream_pos = starts[istream];
        callback_data.decomp_nelem = 0;
        // Set the output buffer to the address of the beginning of this stream.
        callback_data.decompressed = decompressed + istream * n_decode * n_channels;

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
            fprintf(
                stderr,
                "  decoding all samples (n_decode = %ld, stream_size = %ld)\n",
                n_decode,
                stream_size
            );
            success = FLAC__stream_decoder_process_until_end_of_stream(decoder);
            fprintf(stderr, "    success = %d\n", (int)success);
            if (!success) {
                errors |= ERROR_DECODE_PROCESS;
                continue;
            }
        } else {
            // We are decoding a slice of samples.  Seek to the start.
            fprintf(
                stderr,
                "  decoding slice of samples starting at %ld, seeking...\n",
                first_decode
            );
            success = FLAC__stream_decoder_seek_absolute(decoder, first_decode);
            fprintf(stderr, "    success = %d\n", (int)success);
            if (!success) {
                errors |= ERROR_DECODE_PROCESS;
                continue;
            }
            // Process single frames until we have accumulated at least the desired
            // number of output samples.
            int64_t curframe = 0;
            while (
                callback_data.decomp_nelem < n_decode
            ) {
                fprintf(
                    stderr,
                    "  decoding frame %ld, decomp_nelem = %ld, n_decode = %ld\n",
                    curframe,
                    callback_data.decomp_nelem,
                    n_decode
                );
                success = FLAC__stream_decoder_process_single(decoder);
                fprintf(stderr, "    success = %d\n", (int)success);
                if (!success) {
                    errors |= ERROR_DECODE_PROCESS;
                    continue;
                }
                curframe++;
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

    free(decompressed);

    return errors;
}

