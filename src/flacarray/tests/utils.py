# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import unittest

import numpy as np

from ..demo import create_fake_data

from ..utils import (
    int_to_float,
    float_to_int,
)


class UtilsTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_float64(self):
        data_shape = (4, 3, 1000)
        quanta = 1.0e-16
        data, _ = create_fake_data(data_shape, 1.0)

        idata, offsets, gains = float_to_int(data, quanta=quanta, precision=None)
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-15, atol=1e-15):
            print("Failed float64 roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        prec = 5
        idata, offsets, gains = float_to_int(data, quanta=None, precision=prec)
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int(
            data, quanta=None, precision=prec * np.ones(data_shape[:-1])
        )
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        quanta = 1e-5
        idata, offsets, gains = float_to_int(data, quanta=quanta, precision=None)
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int(
            data, quanta=quanta * np.ones(data_shape[:-1]), precision=None
        )
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float64 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

    def test_float32(self):
        data_shape = (4, 3, 1000)
        quanta = 1e-6
        data, _ = create_fake_data(data_shape, 1.0)
        data = data.astype(np.float32)
        idata, offsets, gains = float_to_int(data, quanta=quanta, precision=None)
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-5):
            print("Failed float32 roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        prec = 5
        idata, offsets, gains = float_to_int(data, quanta=None, precision=prec)
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int(
            data, quanta=None, precision=prec * np.ones(data_shape[:-1])
        )
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 precision roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)

        quanta = 1e-5
        idata, offsets, gains = float_to_int(data, quanta=quanta, precision=None)
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
        idata, offsets, gains = float_to_int(
            data, quanta=quanta * np.ones(data_shape[:-1]), precision=None
        )
        check = int_to_float(idata, offsets, gains)
        if not np.allclose(check, data, rtol=1e-5, atol=1e-4):
            print("Failed float32 quanta roundtrip")
            print(f"{check} != {data}", flush=True)
            self.assertTrue(False)
