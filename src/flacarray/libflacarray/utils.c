// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <flacarray.h>
#include <stdio.h>


ArrayUint8 * create_array_uint8(int64_t start_size) {
    ArrayUint8 * ret = (ArrayUint8 *)malloc(sizeof(ArrayUint8));
    if (ret == NULL) {
        return NULL;
    }
    ret->size = 0;
    ret->n_elem = 0;
    ret->data = NULL;
    if (start_size > 0) {
        resize_array_uint8(ret, start_size);
    }
    return ret;
}

void destroy_array_uint8(ArrayUint8 * obj) {
    if (obj != NULL) {
        if (obj->data != NULL) {
            free(obj->data);
        }
        free(obj);
    }
    return;
}

int resize_array_uint8(ArrayUint8 * obj, int64_t new_size) {
    int64_t try_size;
    unsigned char * temp_ptr;
    if (obj != NULL) {
        if (obj->data == NULL) {
            // Not yet allocated
            obj->data = (unsigned char *)malloc(new_size * sizeof(unsigned char));
            if (obj->data == NULL) {
                // Allocation failed, set size to zero.
                obj->size = 0;
                obj->n_elem = 0;
            } else {
                // Allocation worked.
                obj->size = new_size;
                obj->n_elem = new_size;
            }
        } else {
            // Data already allocated, check current size
            if (obj->size >= new_size) {
                // We already have enough space
                obj->n_elem = new_size;
            } else {
                // We need to re-allocate.  Grow our size exponentially
                // to reduce the number of future allocations.
                try_size = obj->size;
                while (try_size < new_size) {
                    try_size *= 2;
                }
                temp_ptr = (unsigned char *)realloc(
                    (void*)(obj->data),
                    try_size * sizeof(unsigned char)
                );
                if (temp_ptr != NULL) {
                    // Realloc success
                    obj->data = temp_ptr;
                    obj->size = try_size;
                    obj->n_elem = new_size;
                }
            }
        }
    } else {
        return ERROR_ALLOC;
    }
    return ERROR_NONE;
}

// Check if the current machine is little endian.
bool is_little_endian() {
    int test = 1;
    if (*((char *)(&test)) == 1) {
        return true;
    } else {
        return false;
    }
}

// Union used to unpack the high and low 32bits from a
// big endian int64.  Note that the FLAC library itself already
// handles endian conversion within each 32bit value, so we don't
// need to do any byte-swapping here.  We are just ensuring that
// the two channels represent the same things (high and low order
// 32bits) across architectures.
//
// Another Note:  we are encoding the high and low 32bits as different
// FLAC channels.  This means that the "sign bit" of the lower 32bit
// value is actually being interpreted as a "normal" bit.  In other
// words, imagine you have a 64bit integer with a value of 2^32.  When
// splitting into two 32bit integers, the high value will be "zero" and
// the low value would be "-0" (sign bit set, but all other bits zero).
typedef union {
    int64_t value;
    struct {
        int32_t high, low;
    };
} int64_hilow;

// If necessary (on big-endian systems), allocate a 32bit buffer
// and fill with swapped interleaved values so that the first channel
// is always the lower-order 32bit value.
int get_interleaved(int64_t n_elem, int64_t const * data, int32_t ** interleaved) {
    if (is_little_endian()) {
        // No separate memory buffer needed
        (*interleaved) = (int32_t *)data;
    } else {
        // Have to allocate a buffer to hold the swapped values.
        (*interleaved) = (int32_t *)malloc(2 * n_elem * sizeof(int32_t));
        if ((*interleaved) == NULL) {
            // Allocation failed
            return ERROR_ALLOC;
        }
    }
    return ERROR_NONE;
}

// Copy data to / from interleaved format

void copy_interleaved_64_to_32(int64_t n_elem, int64_t * input, int32_t * output) {
    int64_hilow * view = (int64_hilow *)input;
    if (! is_little_endian()) {
        for (int64_t i = 0; i < n_elem; ++i) {
            output[2 * i] = view[i].low;
            output[2 * i + 1] = view[i].high;
        }
    }
    return;
}

void copy_interleaved_32_to_64(int64_t n_elem, int32_t * input, int64_t * output) {
    int64_hilow * view = (int64_hilow *)output;
    if (! is_little_endian()) {
        for (int64_t i = 0; i < n_elem; ++i) {
            view[i].low = input[2 * i];
            view[i].high = input[2 * i + 1];
        }
    }
    return;
}

// If the interleaved buffer was previously allocated, free it.
void free_interleaved(int32_t * interleaved) {
    if (! is_little_endian()) {
        // Free buffer
        free(interleaved);
    }
    return;
}

int float32_to_int32(
    float const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * quanta,
    int32_t * output,
    float * offsets,
    float * gains
) {
    // FLAC uses signed integers so the max positive value is 2^31 - 1.
    int32_t flac_max = 2147483647;

    float smin;
    float smax;
    int64_t sindx;
    float sval;
    float stemp;
    float squanta;
    float min_quanta;
    float amp;
    int64_t nquant;
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        smin = input[istream * stream_size];
        smax = input[istream * stream_size];
        for (int64_t isamp = 1; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            sval = input[sindx];
            if (sval < smin) {
                smin = sval;
            }
            if (sval > smax) {
                smax = sval;
            }
        }
        offsets[istream] = 0.5 * (smin + smax);

        // Check the minimum quanta size that can be used without the resulting data
        // overflowing the bit limit.
        if ((smin - offsets[istream]) > (smax - offsets[istream])) {
            amp = 1.01 * (smin - offsets[istream]);
        } else {
            amp = 1.01 * (smax - offsets[istream]);
        }
        min_quanta = amp / flac_max;

        if (quanta == NULL) {
            // We are computing the quanta based on the range of the data.
            squanta = min_quanta;
        } else {
            // We are using a pre-defined quanta per stream.
            squanta = quanta[istream];
            // Commented out, since there might be times when the
            // user wants to truncate the peaks of the data.
            //-----------------------------------------------------
            // if (squanta < min_quanta) {
            //     // The requested quanta is too small
            //     return ERROR_CONVERT_TYPE;
            // }
        }

        // Adjust final offset so that it is a whole number of quanta.
        nquant = (int64_t)(0.5 + offsets[istream] / squanta);
        offsets[istream] = squanta * (double)nquant;

        if (squanta == 0) {
            // This happens if all data is zero and we are computing the quanta
            // from the data.
            gains[istream] = 1.0;
        } else {
            gains[istream] = 1.0 / squanta;
        }

        for (int64_t isamp = 0; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            stemp = input[sindx] - offsets[istream];
            output[sindx] = (int32_t)(gains[istream] * stemp + 0.5);
        }
    }
    return ERROR_NONE;
}

int float64_to_int64(
    double const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * quanta,
    int64_t * output,
    double * offsets,
    double * gains
) {
    // FLAC uses signed integers so the max positive value is 2^63 - 1.
    int64_t flac_max = 9223372036854775807;

    double smin;
    double smax;
    int64_t sindx;
    double sval;
    double stemp;
    double squanta;
    double min_quanta;
    double amp;
    int64_t nquant;
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        smin = input[istream * stream_size];
        smax = input[istream * stream_size];
        for (int64_t isamp = 1; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            sval = input[sindx];
            if (sval < smin) {
                smin = sval;
            }
            if (sval > smax) {
                smax = sval;
            }
        }
        offsets[istream] = 0.5 * (smin + smax);

        // Check the minimum quanta size that can be used without the resulting data
        // overflowing the bit limit.
        if ((smin - offsets[istream]) > (smax - offsets[istream])) {
            amp = 1.01 * (smin - offsets[istream]);
        } else {
            amp = 1.01 * (smax - offsets[istream]);
        }
        min_quanta = amp / flac_max;

        if (quanta == NULL) {
            // We are computing the quanta based on the range of the data.
            squanta = min_quanta;
        } else {
            // We are using a pre-defined quanta per stream.
            squanta = quanta[istream];
            // Commented out, since there might be times when the
            // user wants to truncate the peaks of the data.
            //-----------------------------------------------------
            // if (squanta < min_quanta) {
            //     // The requested quanta is too small
            //     return ERROR_CONVERT_TYPE;
            // }
        }

        // Adjust final offset so that it is a whole number of quanta.
        nquant = (int64_t)(0.5 + offsets[istream] / squanta);
        offsets[istream] = squanta * (double)nquant;

        if (squanta == 0) {
            // This happens if all data is zero and we are computing the quanta
            // from the data.
            gains[istream] = 1.0;
        } else {
            gains[istream] = 1.0 / squanta;
        }

        for (int64_t isamp = 0; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            stemp = input[sindx] - offsets[istream];
            output[sindx] = (int64_t)(gains[istream] * stemp + 0.5);
        }
    }
    return ERROR_NONE;
}

void int64_to_float64(
    int64_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * offsets,
    double const * gains,
    double * output
) {
    int64_t sindx;
    double coeff;
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        coeff = 1.0 / gains[istream];
        for (int64_t isamp = 0; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            output[sindx] = offsets[istream] + coeff * (double)input[sindx];
        }
    }
    return;
}

void int32_to_float32(
    int32_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    float const * offsets,
    float const * gains,
    float * output
) {
    int64_t sindx;
    float coeff;
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        coeff = 1.0 / gains[istream];
        for (int64_t isamp = 0; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            output[sindx] = offsets[istream] + coeff * (float)input[sindx];
        }
    }
    return;
}
