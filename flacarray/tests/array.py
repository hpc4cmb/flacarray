# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import unittest

import numpy as np

from ..array import FlacArray
from ..compress import array_compress
from ..decompress import array_decompress
from ..demo import create_fake_data


class ArrayTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_helpers(self):
        data_shape = (4, 3, 10000)
        leading_shape = data_shape[:-1]
        flatsize = np.prod(data_shape)
        rng = np.random.default_rng()

        n_half = 5
        first = data_shape[-1] // 2 - n_half
        last = data_shape[-1] // 2 + n_half

        # int32 data

        data_i32 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(data_shape)
            .astype(np.int32)
        )

        comp_i32, starts_i32, nbytes_i32, off_i32, gain_i32 = array_compress(data_i32, level=5)
        self.assertTrue(off_i32 is None)
        self.assertTrue(gain_i32 is None)

        check_i32 = array_decompress(comp_i32, data_shape[-1], starts_i32, nbytes_i32)
        self.assertTrue(np.array_equal(check_i32, data_i32))

        check_slc_i32 = array_decompress(
            comp_i32,
            data_shape[-1],
            starts_i32,
            nbytes_i32,
            first_stream_sample=first,
            last_stream_sample=last,
        )
        self.assertTrue(np.array_equal(check_slc_i32, data_i32[:, :, first:last]))

        # int64 data.  First try a case that should work, with all values inside
        # the 32bit range.

        data_i64 = rng.integers(
            low=-(2**30), high=2**29, size=flatsize, dtype=np.int64
        ).reshape(data_shape)

        comp_i64, starts_i64, nbytes_i64, off_i64, gain_i64 = array_compress(data_i64, level=5)
        self.assertTrue(gain_i64 is None)

        check_i64 = array_decompress(
            comp_i64, data_shape[-1], starts_i64, nbytes_i64, stream_offsets=off_i64
        )
        self.assertTrue(np.array_equal(check_i64, data_i64))

        check_slc_i64 = array_decompress(
            comp_i64,
            data_shape[-1],
            starts_i64,
            nbytes_i64,
            stream_offsets=off_i64,
            first_stream_sample=first,
            last_stream_sample=last,
        )
        self.assertTrue(np.array_equal(check_slc_i64, data_i64[:, :, first:last]))

        # Now try a case that should NOT work.

        data_i64 = rng.integers(
            low=-(2**60), high=2**62, size=flatsize, dtype=np.int64
        ).reshape(data_shape)

        try:
            comp_i64, starts_i64, nbytes_i64, off_i64, gain_i64 = array_compress(data_i64, level=5)
            print("Failed to catch truncation of int64 data")
            self.assertTrue(False)
        except RuntimeError:
            pass

        # float32 data

        data_f32 = create_fake_data(data_shape, 1.0).astype(np.float32)
        comp_f32, starts_f32, nbytes_f32, off_f32, gain_f32 = array_compress(data_f32, level=5)
        check_f32 = array_decompress(
            comp_f32,
            data_shape[-1],
            starts_f32,
            nbytes_f32,
            stream_offsets=off_f32,
            stream_gains=gain_f32,
        )
        self.assertTrue(np.allclose(check_f32, data_f32, rtol=1e-5, atol=1e-5))

        check_slc_f32 = array_decompress(
            comp_f32,
            data_shape[-1],
            starts_f32,
            nbytes_f32,
            stream_offsets=off_f32,
            stream_gains=gain_f32,
            first_stream_sample=first,
            last_stream_sample=last,
        )
        self.assertTrue(
            np.allclose(check_slc_f32, data_f32[:, :, first:last], rtol=1e-5, atol=1e-5)
        )

        # float64 data

        data_f64 = create_fake_data(data_shape, 1.0)

        comp_f64, starts_f64, nbytes_f64, off_f64, gain_f64 = array_compress(data_f64, level=5)
        check_f64 = array_decompress(
            comp_f64,
            data_shape[-1],
            starts_f64,
            nbytes_f64,
            stream_offsets=off_f64,
            stream_gains=gain_f64,
        )
        self.assertTrue(np.allclose(check_f64, data_f64, rtol=1e-5, atol=1e-5))

        check_slc_f64 = array_decompress(
            comp_f64,
            data_shape[-1],
            starts_f64,
            nbytes_f64,
            stream_offsets=off_f64,
            stream_gains=gain_f64,
            first_stream_sample=first,
            last_stream_sample=last,
        )
        self.assertTrue(
            np.allclose(check_slc_f64, data_f64[:, :, first:last], rtol=1e-5, atol=1e-5)
        )

    def test_array_memory(self):
        data_shape = (4, 3, 10000)
        data_f64 = create_fake_data(data_shape, 1.0)
        n_half = 5
        first = data_shape[-1] // 2 - n_half
        last = data_shape[-1] // 2 + n_half

        farray = FlacArray.from_array(data_f64)
        check_f64 = farray.to_array()
        self.assertTrue(np.allclose(check_f64, data_f64, rtol=1e-5, atol=1e-5))

        check_slc_f64 = farray.to_array(stream_slice=slice(first, last, 1))
        self.assertTrue(
            np.allclose(check_slc_f64, data_f64[:, :, first:last], rtol=1e-5, atol=1e-5)
        )
