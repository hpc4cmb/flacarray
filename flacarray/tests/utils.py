# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import unittest

import numpy as np

from ..demo import create_fake_data

from ..utils import (
    int64_to_int32,
    int32_to_float,
    float_to_int32,
)


class UtilsTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_int64(self):
        data_shape = (4, 3, 1000)
        leading_shape = data_shape[:-1]
        flatsize = np.prod(data_shape)
        rng = np.random.default_rng()

        input32 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(data_shape)
            .astype(np.int64)
        )
        input64 = rng.integers(
            low=-(2**60), high=2**62, size=flatsize, dtype=np.int64
        ).reshape(data_shape)

        idata, ioff = int64_to_int32(input32)
        check = idata + ioff.reshape(leading_shape + (1,))
        self.assertTrue(np.array_equal(check, input32))

        # Check that we get an error for large integer truncation
        try:
            idata, ioff = int64_to_int32(input64)
            print("Failed to catch 64bit integer truncation")
            self.assertTrue(False)
        except RuntimeError:
            pass

    def test_float64(self):
        data_shape = (4, 3, 1000)
        data = create_fake_data(data_shape, 1.0)

        idata, offsets, gains = float_to_int32(data, quanta=None, precision=None)
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-5):
            print("Failed float64 roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        prec = 5
        idata, offsets, gains = float_to_int32(data, quanta=None, precision=prec)
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int32(
            data, quanta=None, precision=prec * np.ones(data_shape[:-1])
        )
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        quant = 1e-5
        idata, offsets, gains = float_to_int32(data, quanta=quant, precision=None)
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int32(
            data, quanta=quant * np.ones(data_shape[:-1]), precision=None
        )
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

    def test_float32(self):
        data_shape = (4, 3, 1000)
        data = create_fake_data(data_shape, 1.0).astype(np.float32)
        idata, offsets, gains = float_to_int32(data, quanta=None, precision=None)
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-5):
            print("Failed float32 roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        prec = 5
        idata, offsets, gains = float_to_int32(data, quanta=None, precision=prec)
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int32(
            data, quanta=None, precision=prec * np.ones(data_shape[:-1])
        )
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        quant = 1e-5
        idata, offsets, gains = float_to_int32(data, quanta=quant, precision=None)
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int32(
            data, quanta=quant * np.ones(data_shape[:-1]), precision=None
        )
        check = int32_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
