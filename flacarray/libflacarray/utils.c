// Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include "flacarray.h"


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

ArrayInt32 * create_array_int32(int64_t start_size) {
    ArrayInt32 * ret = (ArrayInt32 *)malloc(sizeof(ArrayInt32));
    if (ret == NULL) {
        return NULL;
    }
    (*ret).size = 0;
    (*ret).n_elem = 0;
    (*ret).data = NULL;
    if (start_size > 0) {
        resize_array_int32(ret, start_size);
    }
    return ret;
}

void destroy_array_int32(ArrayInt32 * obj) {
    if (obj != NULL) {
        if (obj->data != NULL) {
            free(obj->data);
        }
        free(obj);
    }
    return;
}

int resize_array_int32(ArrayInt32 * obj, int64_t new_size) {
    int64_t try_size;
    int32_t * temp_ptr;
    if (obj != NULL) {
        if (obj->data == NULL) {
            // Not yet allocated
            obj->data = (int32_t *)malloc(new_size * sizeof(int32_t));
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
                temp_ptr = (int32_t *)realloc(
                    (void*)(obj->data),
                    try_size * sizeof(int32_t)
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


int int64_to_int32(
    int64_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    int32_t * output,
    int64_t * offsets
) {
    // FLAC uses an extra bit, so +/- 2^30 is the max range.
    int64_t flac_max = 1073741824;

    int64_t smin;
    int64_t smax;
    int64_t sindx;
    int64_t sval;
    int64_t stemp;
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
        offsets[istream] = (int64_t)((0.5 * (double)(smin + smax)) + 0.5);
        for (int64_t isamp = 0; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            stemp = input[sindx] - offsets[istream];
            if ((stemp > flac_max) || (stemp < -flac_max)) {
                return ERROR_CONVERT_TYPE;
            } else {
                output[sindx] = (int32_t)stemp;
            }
        }
    }
    return ERROR_NONE;
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
    // FLAC uses an extra bit, so +/- 2^30 is the max range.
    int64_t flac_max = 1073741824;

    float smin;
    float smax;
    int64_t sindx;
    float sval;
    float stemp;
    float squanta;
    float min_quanta;
    float amp;
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
        // overflowing the 2^30 bit limit.
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


int float64_to_int32(
    double const * input,
    int64_t n_stream,
    int64_t stream_size,
    double const * quanta,
    int32_t * output,
    double * offsets,
    double * gains
) {
    // FLAC uses an extra bit, so +/- 2^30 is the max range.
    int64_t flac_max = 1073741824;

    double smin;
    double smax;
    int64_t sindx;
    double sval;
    double stemp;
    double squanta;
    double min_quanta;
    double amp;
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
        // overflowing the 2^30 bit limit.
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


void int32_to_int64(
    int32_t const * input,
    int64_t n_stream,
    int64_t stream_size,
    int64_t const * offsets,
    int64_t * output
) {
    int64_t sindx;
    for (int64_t istream = 0; istream < n_stream; ++istream) {
        for (int64_t isamp = 0; isamp < stream_size; ++isamp) {
            sindx = istream * stream_size + isamp;
            output[sindx] = offsets[istream] + (int64_t)input[sindx];
        }
    }
    return;
}


void int32_to_float64(
    int32_t const * input,
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
