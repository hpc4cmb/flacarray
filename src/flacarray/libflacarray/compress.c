// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <stdbool.h>
#include <string.h>

#include <flacarray.h>


// Callback function, called by the encoder for each chunk
// of data.
FLAC__StreamEncoderWriteStatus enc_write_callback(
    const FLAC__StreamEncoder * encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    uint32_t samples,
    uint32_t current_frame,
    void * client_data
) {
    enc_callback_data * data = (enc_callback_data *)client_data;

    int64_t last = data->last_stream;
    int64_t cur = data->cur_stream;
    int64_t * stream_offsets = data->stream_offsets;
    ArrayUint8 * comp = data->compressed;

    // See if we have started a new stream, and update the starting location.
    if (cur != last) {
        // We are starting a new stream.  Record the current starting offset.
        if (last < 0) {
            // This is the first stream
            stream_offsets[cur] = 0;
        } else {
            stream_offsets[cur] = comp->n_elem;
        }
        data->last_stream = cur;
    }

    // Initialize the buffer for this stream if it is the first call.  Otherwise
    // resize as needed.
    int64_t elems;
    int64_t new_elems;
    if (comp == NULL) {
        elems = 0;
        data->compressed = create_array_uint8(bytes);
        comp = data->compressed;
        new_elems = bytes;
    } else {
        elems = comp->n_elem;
        new_elems = elems + bytes;
        resize_array_uint8(comp, new_elems);
    }

    // Copy bytes into place
    memcpy(
        (void*)(comp->data + elems),
        (void*)buffer,
        bytes
    );

    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}


// Callback function, called by the encoder for each chunk
// of data.
FLAC__StreamEncoderWriteStatus enc_threaded_write_callback(
    const FLAC__StreamEncoder * encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    uint32_t samples,
    uint32_t current_frame,
    void * client_data
) {
    enc_threaded_callback_data * data = (enc_threaded_callback_data *)client_data;

    int64_t cur = data->cur_stream;
    ArrayUint8 * comp = data->compressed[cur];

    // Initialize the buffer for this stream if it is the first call.  Otherwise
    // resize as needed.
    int64_t elems;
    int64_t new_elems;
    if (comp == NULL) {
        elems = 0;
        data->compressed[cur] = create_array_uint8(bytes);
        comp = data->compressed[cur];
        new_elems = bytes;
    } else {
        elems = comp->n_elem;
        new_elems = elems + bytes;
        resize_array_uint8(comp, new_elems);
    }

    // Copy bytes into place
    memcpy(
        (void*)(comp->data + elems),
        (void*)buffer,
        bytes
    );

    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}


// Helper function to free an array of buffers
void free_compressed_buffers(ArrayUint8 ** buffers, int64_t n_stream) {
    if (buffers != NULL) {
        for (int64_t istream = 0; istream < n_stream; ++istream) {
            if (buffers[istream] != NULL) {
                destroy_array_uint8(buffers[istream]);
            }
        }
        free(buffers);
    }
    return;
}


// Main encode functions.  A newly allocated buffer of bytes is returned along
// with the starting byte in this buffer for each of the streams.  This function
// requires that the N-dimensional array is contiguous in memory and is treated
// as a flat-packed array.  If any errors occur, the processing stops, an
// attempt is made to free any buffers that were allocated, and an error code is
// returned which is a bitwise OR of the errors on all threads.

// NOTE:  libFLAC >= 1.5.0 natively supports threaded compression, but this is not
// enabled by default.  We could evaluate the performance of that, but would require
// a check on the version being used.

// Unthreaded version.  No need for thread-local buffers, so this is often faster.
int encode(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t n_channels,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
) {
    // Check input parameters
    if (level > 8) {
        return ERROR_INVALID_LEVEL;
    }
    if (n_stream == 0) {
        return ERROR_ZERO_NSTREAM;
    }
    if (stream_size == 0) {
        return ERROR_ZERO_STREAMSIZE;
    }

    // Zero out return values to start.
    (*n_bytes) = 0;
    (*bytes) = NULL;
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        starts[istream] = 0;
    }

    // This tracks the failures.
    int errors = ERROR_NONE;

    // Encoder variables.
    bool success;
    FLAC__StreamEncoderInitStatus status;
    FLAC__StreamEncoder * encoder;

    // Create callback data
    enc_callback_data callback_data;
    callback_data.last_stream = -1;
    callback_data.stream_offsets = starts;
    callback_data.compressed = NULL;

    for (int64_t istream = 0; istream < n_stream; ++istream) {
        if (errors != ERROR_NONE) {
            // We already had a failure, skip over remaining loop iterations
            continue;
        }
        // Set the current stream in the callback data
        callback_data.cur_stream = istream;

        // Create encoder and set parameters.
        encoder = FLAC__stream_encoder_new();
        success = FLAC__stream_encoder_set_compression_level(encoder, level);
        if (! success) {
            errors |= ERROR_ENCODE_SET_COMP_LEVEL;
            continue;
        }
        success = FLAC__stream_encoder_set_blocksize(encoder, 0);
        if (! success) {
            errors |= ERROR_ENCODE_SET_BLOCK_SIZE;
            continue;
        }
        success = FLAC__stream_encoder_set_channels(encoder, n_channels);
        if (!success) {
            errors |= ERROR_ENCODE_SET_CHANNELS;
            continue;
        }
        success = FLAC__stream_encoder_set_bits_per_sample(encoder, 32);
        if (!success) {
            errors |= ERROR_ENCODE_SET_BPS;
            continue;
        }

        // Initialize our encoder with our callback function and data.
        status = FLAC__stream_encoder_init_stream(
            encoder,
            enc_write_callback,
            NULL,
            NULL,
            NULL,
            (void *)&callback_data
        );
        if (status != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
            errors |= ERROR_ENCODE_INIT;
            continue;
        }

        // Encode this stream.
        success = FLAC__stream_encoder_process_interleaved(
            encoder,
            &(data[istream * stream_size * n_channels]),
            stream_size
        );
        if (!success) {
            errors |= ERROR_ENCODE_PROCESS;
            continue;
        }

        // Encoder cleanup.
        success = FLAC__stream_encoder_finish(encoder);
        if (!success) {
            errors |= ERROR_ENCODE_FINISH;
            continue;
        }
        FLAC__stream_encoder_delete(encoder);
    }

    if (errors != ERROR_NONE) {
        // Clean up and exit
        destroy_array_uint8(callback_data.compressed);
        return errors;
    }

    // Total number of bytes.  The starting offsets were computed on the
    // fly during the encode whenever switching to the next stream.
    (*n_bytes) = callback_data.compressed->n_elem;

    // Allocate the output bytes
    (*bytes) = (unsigned char *)malloc((*n_bytes));
    if ((*bytes) == NULL) {
        // Allocation failed.
        destroy_array_uint8(callback_data.compressed);
        errors |= ERROR_ALLOC;
        return errors;
    }

    // Copy the accumulated byte buffer into the output.
    memcpy(
        (void*)(*bytes),
        (void*)(callback_data.compressed->data),
        (*n_bytes) * sizeof(unsigned char)
    );

    // Cleanup callback structure
    destroy_array_uint8(callback_data.compressed);

    return errors;
}


// Threaded version
int encode_threaded(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t n_channels,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
) {
    // Check input parameters
    if (level > 8) {
        return ERROR_INVALID_LEVEL;
    }
    if (n_stream == 0) {
        return ERROR_ZERO_NSTREAM;
    }
    if (stream_size == 0) {
        return ERROR_ZERO_STREAMSIZE;
    }

    // Zero out return values to start.
    (*n_bytes) = 0;
    (*bytes) = NULL;

    // Allocate an array of buffer pointers, one per stream.  This will be
    // populated by individual threads and then copied into the final output
    // buffer at the end.
    ArrayUint8 ** buffers = (ArrayUint8 **)malloc(n_stream * sizeof(ArrayUint8 *));
    if (buffers == NULL) {
        // Allocation failed
        return ERROR_ALLOC;
    } else {
        for (int64_t istream = 0; istream < n_stream; ++istream) {
            buffers[istream] = NULL;
        }
    }

    // This tracks the failures across all threads.
    int errors = ERROR_NONE;

    #pragma omp parallel reduction(|:errors)
    {
        // Thread-local encoder variables.
        bool success;
        FLAC__StreamEncoderInitStatus status;
        FLAC__StreamEncoder * encoder;

        // Create thread-local callback data
        enc_threaded_callback_data callback_data;
        callback_data.n_stream = n_stream;
        callback_data.compressed = buffers;

        #pragma omp for schedule(static)
        for (int64_t istream = 0; istream < n_stream; ++istream) {
            if (errors != ERROR_NONE) {
                // We already had a failure, skip over remaining loop iterations
                continue;
            }
            // Set the current stream in the callback data
            callback_data.cur_stream = istream;

            // Create encoder and set parameters.
            encoder = FLAC__stream_encoder_new();
            success = FLAC__stream_encoder_set_compression_level(encoder, level);
            if (! success) {
                errors |= ERROR_ENCODE_SET_COMP_LEVEL;
                continue;
            }
            success = FLAC__stream_encoder_set_blocksize(encoder, 0);
            if (! success) {
                errors |= ERROR_ENCODE_SET_BLOCK_SIZE;
                continue;
            }
            success = FLAC__stream_encoder_set_channels(encoder, n_channels);
            if (!success) {
                errors |= ERROR_ENCODE_SET_CHANNELS;
                continue;
            }
            success = FLAC__stream_encoder_set_bits_per_sample(encoder, 32);
            if (!success) {
                errors |= ERROR_ENCODE_SET_BPS;
                continue;
            }

            // Initialize our encoder with our callback function and data.
            status = FLAC__stream_encoder_init_stream(
                encoder,
                enc_threaded_write_callback,
                NULL,
                NULL,
                NULL,
                (void *)&callback_data
            );
            if (status != FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
                errors |= ERROR_ENCODE_INIT;
                continue;
            }

            // Encode this stream.
            success = FLAC__stream_encoder_process_interleaved(
                encoder,
                &(data[istream * stream_size * n_channels]),
                stream_size
            );
            if (!success) {
                errors |= ERROR_ENCODE_PROCESS;
                continue;
            }

            // Encoder cleanup.
            success = FLAC__stream_encoder_finish(encoder);
            if (!success) {
                errors |= ERROR_ENCODE_FINISH;
                continue;
            }
            FLAC__stream_encoder_delete(encoder);
        }
    }

    if (errors != ERROR_NONE) {
        // Clean up and exit
        free_compressed_buffers(buffers, n_stream);
        return errors;
    }

    // Compute the starts of each stream in the final bytes buffer
    // and the total number of output bytes.
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        if (buffers[istream] == NULL) {
            // One of the streams was never processed.  This is an error.
            errors |= ERROR_ENCODE_COLLECT;
            free_compressed_buffers(buffers, n_stream);
            return errors;
        }
        starts[istream] = (*n_bytes);
        (*n_bytes) += buffers[istream]->n_elem;
    }

    // Allocate the output bytes
    (*bytes) = (unsigned char *)malloc((*n_bytes));
    if ((*bytes) == NULL) {
        // Allocation failed.
        free_compressed_buffers(buffers, n_stream);
        errors |= ERROR_ALLOC;
        return errors;
    }

    // Copy the per-stream buffers into the output.
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        memcpy(
            (void*)((*bytes) + starts[istream]),
            (void*)(buffers[istream]->data),
            buffers[istream]->n_elem * sizeof(unsigned char)
        );
    }

    // Cleanup
    free_compressed_buffers(buffers, n_stream);

    return errors;
}


// Helper wrappers for 32bit and 64bit integers.

int encode_i32(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
) {
    return encode(
        data,
        n_stream,
        stream_size,
        1,
        level,
        n_bytes,
        starts,
        bytes
    );
}

int encode_i32_threaded(
    int32_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
) {
    return encode_threaded(
        data,
        n_stream,
        stream_size,
        1,
        level,
        n_bytes,
        starts,
        bytes
    );
}

int encode_i64(
    int64_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
) {
    int64_t n_elem = n_stream * stream_size;
    int32_t * interleaved;
    int err = get_interleaved(n_elem, data, &interleaved);
    if (err != ERROR_NONE) {
        return err;
    }
    copy_interleaved_64_to_32(n_elem, data, interleaved);
    err = encode(
        interleaved,
        n_stream,
        stream_size,
        2,
        level,
        n_bytes,
        starts,
        bytes
    );
    free_interleaved(interleaved);
    return err;
}

int encode_i64_threaded(
    int64_t * const data,
    int64_t n_stream,
    int64_t stream_size,
    uint32_t level,
    int64_t * n_bytes,
    int64_t * starts,
    unsigned char ** bytes
) {
    int64_t n_elem = n_stream * stream_size;
    int32_t * interleaved;
    int err = get_interleaved(n_elem, data, &interleaved);
    if (err != ERROR_NONE) {
        return err;
    }
    copy_interleaved_64_to_32(n_elem, data, interleaved);
    err = encode_threaded(
        interleaved,
        n_stream,
        stream_size,
        2,
        level,
        n_bytes,
        starts,
        bytes
    );
    free_interleaved(interleaved);
    return err;
}
