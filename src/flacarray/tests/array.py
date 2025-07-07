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
from ..mpi import use_mpi, MPI
from ..utils import float_to_int, int_to_float


class ArrayTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if use_mpi:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = None

    def test_helpers(self):
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank
        for data_shape in [
            (4, 3, 1000),
            (10000,),
        ]:
            shpstr = "x".join([f"{x}" for x in data_shape])
            for dt, dtstr, sigma, quant in [
                (np.dtype(np.int32), "i32", None, None),
                (np.dtype(np.int64), "i64", None, None),
                (np.dtype(np.float32), "f32", 1.0, 1.0e-6),
                (np.dtype(np.float64), "f64", 1.0, 1.0e-7),
            ]:
                ftol = 1.0e-5
                if quant is not None:
                    ftol = 10.0 * quant
                # Run identical tests on all processes (no MPI).
                input, _ = create_fake_data(
                    data_shape, sigma=sigma, dtype=dt, comm=None
                )

                n_half = 5
                first = data_shape[-1] // 2 - n_half
                last = data_shape[-1] // 2 + n_half

                comp, starts, nbytes, off, gain = array_compress(
                    input, level=5, quanta=quant
                )
                if dt == np.dtype(np.int32) or dt == np.dtype(np.int64):
                    # Integer inputs, gains and offsets should be None
                    self.assertTrue(off is None)
                    self.assertTrue(gain is None)

                # Decompress full array and compare

                if dt == np.dtype(np.int64) or dt == np.dtype(np.float64):
                    check = array_decompress(
                        comp,
                        data_shape[-1],
                        starts,
                        nbytes,
                        stream_offsets=off,
                        stream_gains=gain,
                        is_int64=True,
                    )
                else:
                    check = array_decompress(
                        comp,
                        data_shape[-1],
                        starts,
                        nbytes,
                        stream_offsets=off,
                        stream_gains=gain,
                        is_int64=False,
                    )

                if dt == np.dtype(np.int32) or dt == np.dtype(np.int64):
                    local_fail = not np.array_equal(check, input)
                elif dt == np.dtype(np.float32):
                    local_fail = not np.allclose(check, input, atol=ftol)
                else:
                    local_fail = not np.allclose(check, input, atol=ftol)
                if self.comm is not None:
                    fail = self.comm.allreduce(local_fail, op=MPI.SUM)
                else:
                    fail = local_fail
                if fail:
                    print(f"input_{dtstr}_{shpstr}[{rank}] = {input}", flush=True)
                    print(f"check_{dtstr}_{shpstr}[{rank}] = {check}", flush=True)
                    print(f"FAIL on {dtstr} roundtrip", flush=True)
                    self.assertTrue(False)

                # Decompress a slice and compare

                if dt == np.dtype(np.int64) or dt == np.dtype(np.float64):
                    check = array_decompress(
                        comp,
                        data_shape[-1],
                        starts,
                        nbytes,
                        stream_offsets=off,
                        stream_gains=gain,
                        first_stream_sample=first,
                        last_stream_sample=last,
                        is_int64=True,
                    )
                else:
                    check = array_decompress(
                        comp,
                        data_shape[-1],
                        starts,
                        nbytes,
                        stream_offsets=off,
                        stream_gains=gain,
                        first_stream_sample=first,
                        last_stream_sample=last,
                        is_int64=False,
                    )

                slc = [slice(0, x) for x in input.shape]
                slc[-1] = slice(first, last)
                slc = tuple(slc)

                if dt == np.dtype(np.int32) or dt == np.dtype(np.int64):
                    local_fail = not np.array_equal(check, input[slc])
                elif dt == np.dtype(np.float32):
                    local_fail = not np.allclose(check, input[slc], atol=ftol)
                else:
                    local_fail = not np.allclose(check, input[slc], atol=ftol)
                if self.comm is not None:
                    fail = self.comm.allreduce(local_fail, op=MPI.SUM)
                else:
                    fail = local_fail
                if fail:
                    print(f"check_{dtstr}_{shpstr}[{rank}] = {check}", flush=True)
                    print(f"input_{dtstr}_{shpstr}[{rank}] = {input[slc]}", flush=True)
                    print(f"FAIL on {dtstr} roundtrip slice {slc}", flush=True)
                    self.assertTrue(False)

    def test_array_memory(self):
        data_shape = (4, 3, 10000)
        quanta = 1.0e-16
        data_f64, _ = create_fake_data(data_shape, 1.0)
        n_half = 5
        first = data_shape[-1] // 2 - n_half
        last = data_shape[-1] // 2 + n_half

        farray = FlacArray.from_array(data_f64, quanta=quanta)
        check_f64 = farray.to_array()
        self.assertTrue(np.allclose(check_f64, data_f64, rtol=1e-15, atol=1e-15))

        check_slc_f64 = farray.to_array(stream_slice=slice(first, last, 1))
        self.assertTrue(
            np.allclose(
                check_slc_f64, data_f64[:, :, first:last], rtol=1e-15, atol=1e-15
            )
        )

    def test_slicing_shape(self):
        data_shape = (4, 3, 10, 100)
        flatsize = np.prod(data_shape)
        rng = np.random.default_rng()
        data_i32 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(data_shape)
            .astype(np.int32)
        )

        farray = FlacArray.from_array(data_i32)

        # Try some slices and verify expected result shape.
        for dslc in [
            (slice(0)),
            (slice(1, 3)),
            (slice(3, 1)),
            (slice(3, 1, -1)),
            (1, 2, 5, 50),
            (1, 2, 5),
            (2, slice(0, 1, 1), slice(0, 1, 1), slice(None)),
            (1, slice(1, 3, 1), slice(6, 8, 1), 50),
            (slice(1, 3, 1), 2, slice(6, 8, 1), slice(60, 80, 1)),
            (2, 1, slice(2, 8, 2), slice(80, 120, 1)),
            (2, 1, slice(2, 8, 2), slice(80, None)),
            (2, 1, slice(2, 8, 2), slice(None, 10)),
        ]:
            # Slice of the original numpy array
            check = data_i32[dslc]
            # Slice of the FlacArray
            fcheck = farray[dslc]

            # Compare the shapes
            if fcheck.shape != check.shape:
                print(
                    f"Array[{dslc}] shape: {fcheck.shape} != {check.shape}",
                    flush=True,
                )
                raise RuntimeError("Failed slice shape check")

    def test_slicing_1D(self):
        data_shape = (10000,)
        flatsize = np.prod(data_shape)
        rng = np.random.default_rng()
        data_i32 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(data_shape)
            .astype(np.int32)
        )

        farray = FlacArray.from_array(data_i32)

        # Try some slices and verify expected result shape.
        for dslc in [(slice(0)), (slice(1, 3)), (100,)]:
            # Slice of the original numpy array
            check = data_i32[dslc]
            # Slice of the FlacArray
            fcheck = farray[dslc]

            # Compare the shapes
            if fcheck.shape != check.shape:
                print(
                    f"Array[{dslc}] shape: {fcheck.shape} != {check.shape}",
                    flush=True,
                )
                raise RuntimeError("Failed slice shape check")

    def test_quantization(self):
        data_shape = (1, 1000)
        for dt, dtstr, sigma, quant in [
            (np.dtype(np.float32), "f32", 1.0, 1.0e-3),
            (np.dtype(np.float64), "f64", 1.0, 1.0e-3),
        ]:
            for dc in [0.0, -0.5, 0.5, -10.0, 10.0, -10.51, -10.4]:
                # Run identical tests on all processes (no MPI).
                input, _ = create_fake_data(
                    data_shape, sigma=sigma, dtype=dt, comm=None, dc_sigma=None
                )
                original = input + dc

                # First test that roundtrip conversion of this random data to integer
                # and back results in errors that are less than half the quantization
                # value.

                data_int, data_offset, data_gain = float_to_int(original, quanta=quant)
                output = int_to_float(data_int, data_offset, data_gain)
                residual = np.absolute(output - original)
                max_resid = np.amax(residual)
                if max_resid > 0.5 * quant:
                    msg = f"FAIL: Quantization of {dtstr} with quant={quant}, "
                    msg += f"offset={dc} has max absolute error of {max_resid} "
                    msg += f"which is larger than 0.5 * quant ({0.5 * quant})"
                    print(msg, flush=True)
                    self.assertTrue(False)

                # Next, pre-truncate the input random floating point data to the nearest
                # quanta value.  The resulting quantized data should compress losslessly
                # up to the machine precision of the dtype we are using across the
                # dynamic range of the data.
                mach_prec = np.finfo(dt).eps

                quantized = np.array(original / quant, dtype=np.int64).astype(dt)
                quantized *= quant
                quantized_range = 2 * np.amax(np.absolute(quantized))
                q_err = quantized_range * mach_prec

                data_int, data_offset, data_gain = float_to_int(quantized, quanta=quant)
                output = int_to_float(data_int, data_offset, data_gain)
                residual = np.absolute(output - quantized)
                max_resid = np.amax(residual)
                if max_resid > q_err:
                    msg = f"FAIL: Quantization of pre-truncated {dtstr} with "
                    msg += f"quant={quant}, offset={dc} has max absolute error of"
                    msg += f" {max_resid} which is larger than expected for dtype"
                    msg += f" and data range ({q_err})"
                    print(msg, flush=True)
                    self.assertTrue(False)
