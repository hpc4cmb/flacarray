// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <stdio.h>
#include <time.h>

#include "flacarray.h"

int main(int argc, char *argv[]) {
    int64_t n_streams = 10;
    int64_t stream_len = 1000000;
    int64_t input_bytes = n_streams * stream_len * sizeof(int32_t);
    uint32_t level = 5;

    int64_t n_bytes;
    uint8_t *compressed;
    int64_t *stream_starts;
    int64_t *stream_nbytes;

    // Set up random number generator
    int64_t n_state = 64;
    char state[n_state];
    char *previous_state = initstate(123456, state, n_state);
    previous_state = setstate(state);

    // Alloc input buffer
    int32_t *data = (int32_t *)malloc(n_streams * stream_len * sizeof(int32_t));
    if (data == NULL) {
        fprintf(stderr, "Failed to allocate input data\n");
    }

    // Allocate the buffer of starts into the bytestream
    stream_starts = (int64_t *)malloc(n_streams * sizeof(int64_t));
    if (stream_starts == NULL) {
        fprintf(stderr, "Failed to allocate starts\n");
    }

    // Allocate the buffer of bytes per stream
    stream_nbytes = (int64_t *)malloc(n_streams * sizeof(int64_t));
    if (stream_nbytes == NULL) {
        fprintf(stderr, "Failed to allocate stream nbytes\n");
    }

    // Fill with randoms
    long temp;
    for (int64_t istream = 0; istream < n_streams; ++istream) {
        for (int64_t isamp = 0; isamp < stream_len; ++isamp) {
            temp = random();
            data[istream * stream_len + isamp] = (int32_t)(temp);
        }
    }

    clock_t start = clock();
    clock_t diff;

    // Encode to bytes
    int status = encode(
        data,
        n_streams,
        stream_len,
        level,
        &n_bytes,
        stream_starts,
        &compressed);

    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;

    fprintf(stderr, "Encoded %ld streams of %ld integers (%ld bytes) into %ld bytes, status = %d\n", n_streams, stream_len, input_bytes, n_bytes, status);
    fprintf(stderr, "  CPU time:  %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    free(compressed);

    start = clock();

    // Encode to bytes
    status = encode_threaded(
        data,
        n_streams,
        stream_len,
        level,
        &n_bytes,
        stream_starts,
        &compressed);

    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;

    fprintf(stderr, "Encoded (threaded) %ld streams of %ld integers (%ld bytes) into %ld bytes, status = %d\n", n_streams, stream_len, input_bytes, n_bytes, status);
    fprintf(stderr, "  CPU time:  %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    // Compute stream nbytes
    for (int64_t istream = 0; istream < n_streams - 1; ++istream) {
        stream_nbytes[istream] = stream_starts[istream + 1] - stream_starts[istream];
    }
    stream_nbytes[n_streams - 1] = n_bytes - stream_starts[n_streams - 1];

    int32_t *decompressed = (int32_t *)malloc(
        n_streams * stream_len * sizeof(int32_t));
    if (decompressed == NULL) {
        fprintf(stderr, "Failed to allocate decompressed data\n");
    }

    int64_t first_sample = -1;
    int64_t last_sample = -1;

    start = clock();
    status = decode(
        compressed,
        stream_starts,
        stream_nbytes,
        n_streams,
        stream_len,
        first_sample,
        last_sample,
        decompressed,
        false);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;

    fprintf(stderr, "Decoded %ld streams of %ld integers, status = %d\n", n_streams, stream_len, status);
    fprintf(stderr, "  CPU time:  %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    start = clock();
    status = decode(
        compressed,
        stream_starts,
        stream_nbytes,
        n_streams,
        stream_len,
        first_sample,
        last_sample,
        decompressed,
        true);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;

    fprintf(stderr, "Decoded (with threads) %ld streams of %ld integers, status = %d\n", n_streams, stream_len, status);
    fprintf(stderr, "  CPU time:  %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    // Verify
    int64_t elem;
    for (int64_t istream = 0; istream < n_streams; ++istream) {
        for (int64_t isamp = 0; isamp < stream_len; ++isamp) {
            elem = istream * stream_len + isamp;
            if (data[elem] != decompressed[elem]) {
                fprintf(stderr,
                    "FAIL stream %ld, sample %ld:  %d != %d\n",
                    istream, isamp, decompressed[elem], data[elem]);
            }
        }
    }
    fprintf(stderr, "SUCCESS\n");

    free(decompressed);

    // Now try decoding a slice

    first_sample = (int64_t)(stream_len / 2) - 5;
    last_sample = (int64_t)(stream_len / 2) + 5;

    first_sample = 1;
    last_sample = 6;

    int64_t n_decode = last_sample - first_sample;

    decompressed = (int32_t *)malloc(
        n_streams * n_decode * sizeof(int32_t));
    if (decompressed == NULL) {
        fprintf(stderr, "Failed to allocate decompressed data\n");
    }

    start = clock();
    status = decode(
        compressed,
        stream_starts,
        stream_nbytes,
        n_streams,
        stream_len,
        first_sample,
        last_sample,
        decompressed,
        false);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;

    fprintf(stderr, "Decoded %ld streams with slice of %ld integers, status = %d\n", n_streams, n_decode, status);
    fprintf(stderr, "  CPU time:  %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    start = clock();
    status = decode(
        compressed,
        stream_starts,
        stream_nbytes,
        n_streams,
        stream_len,
        first_sample,
        last_sample,
        decompressed,
        true);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;

    fprintf(stderr, "Decoded (with threads) %ld streams with slice of %ld integers, status = %d\n", n_streams, n_decode, status);
    fprintf(stderr, "  CPU time:  %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

    // Verify
    int64_t input_elem;
    int64_t output_elem;
    for (int64_t istream = 0; istream < n_streams; ++istream) {
        for (int64_t isamp = first_sample; isamp < last_sample; ++isamp) {
            input_elem = istream * stream_len + isamp;
            output_elem = istream * n_decode + (isamp - first_sample);
            if (data[input_elem] != decompressed[output_elem]) {
                fprintf(stderr,
                    "FAIL stream %ld, sample %ld:  %d != %d\n",
                    istream, isamp, decompressed[output_elem], data[input_elem]);
            }
        }
    }
    fprintf(stderr, "SUCCESS\n");

    free(decompressed);

    free(compressed);
    free(stream_starts);
    free(stream_nbytes);

    free(data);

    return 0;
}
