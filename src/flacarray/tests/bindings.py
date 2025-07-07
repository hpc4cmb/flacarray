# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import unittest

import numpy as np

from ..libflacarray import (
    wrap_encode_i32,
    wrap_encode_i32_threaded,
    wrap_encode_i64,
    wrap_encode_i64_threaded,
    wrap_decode_i32,
    wrap_decode_i64,
    encode_flac,
    decode_flac,
)
from ..demo import create_fake_data


class BindingsTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_wrappers_i32(self):
        n_streams = 3
        stream_len = 10000
        level = 5

        flatsize = n_streams * stream_len

        rng = np.random.default_rng()
        data = rng.integers(low=-(2**31), high=2**31, size=flatsize, dtype=np.int32)

        # Set a few values to extrema, to test correct handling
        data[0] = -2147483647
        data[1] = 2147483647

        compressed, stream_starts, stream_nbytes = wrap_encode_i32_threaded(
            data, n_streams, stream_len, level
        )

        output = wrap_decode_i32(
            compressed,
            stream_starts,
            stream_nbytes,
            n_streams,
            stream_len,
            -1,
            -1,
            True,
        )

        for istream in range(n_streams):
            for isamp in range(stream_len):
                if (
                    output[istream * stream_len + isamp]
                    != data[istream * stream_len + isamp]
                ):
                    msg = f"FAIL [{istream},{isamp}]: "
                    msg += f"{output[istream * stream_len + isamp]} "
                    msg += f"!= {data[istream * stream_len + isamp]}"
                    print(msg)
                    self.assertTrue(False)

        # Now testing with sample slices
        first = (stream_len // 2) - 5
        last = (stream_len // 2) + 5
        n_decode = last - first

        output_slc = wrap_decode_i32(
            compressed,
            stream_starts,
            stream_nbytes,
            n_streams,
            stream_len,
            first,
            last,
            True,
        )
        for istream in range(n_streams):
            for isamp in range(n_decode):
                if (
                    output_slc[istream * n_decode + isamp]
                    != data[istream * stream_len + isamp + first]
                ):
                    msg = f"FAIL [{istream},{isamp}]: "
                    msg += f"{output_slc[istream * n_decode + isamp]} "
                    msg += f"!= {data[istream * stream_len + isamp + first]}"
                    print(msg)
                    self.assertTrue(False)

    def test_wrappers_i64(self):
        n_streams = 3
        stream_len = 10000
        level = 5

        flatsize = n_streams * stream_len

        rng = np.random.default_rng()
        data = rng.integers(low=-(2**63), high=2**63, size=flatsize, dtype=np.int64)

        # Set a few values to extrema, to test correct handling
        data[0] = -9223372036854775807
        data[1] = 9223372036854775807
        data[2] = 4294967296
        data[3] = -4294967296

        compressed, stream_starts, stream_nbytes = wrap_encode_i64_threaded(
            data, n_streams, stream_len, level
        )

        output = wrap_decode_i64(
            compressed,
            stream_starts,
            stream_nbytes,
            n_streams,
            stream_len,
            -1,
            -1,
            True,
        )

        for istream in range(n_streams):
            for isamp in range(stream_len):
                if (
                    output[istream * stream_len + isamp]
                    != data[istream * stream_len + isamp]
                ):
                    msg = f"FAIL [{istream},{isamp}]: "
                    msg += f"{output[istream * stream_len + isamp]} "
                    msg += f"!= {data[istream * stream_len + isamp]}"
                    print(msg)
                    self.assertTrue(False)

        # Now testing with sample slices
        first = (stream_len // 2) - 5
        last = (stream_len // 2) + 5
        n_decode = last - first

        output_slc = wrap_decode_i64(
            compressed,
            stream_starts,
            stream_nbytes,
            n_streams,
            stream_len,
            first,
            last,
            True,
        )
        for istream in range(n_streams):
            for isamp in range(n_decode):
                if (
                    output_slc[istream * n_decode + isamp]
                    != data[istream * stream_len + isamp + first]
                ):
                    msg = f"FAIL [{istream},{isamp}]: "
                    msg += f"{output_slc[istream * n_decode + isamp]} "
                    msg += f"!= {data[istream * stream_len + isamp + first]}"
                    print(msg)
                    self.assertTrue(False)

    def test_roundtrip(self):
        level = 5
        for data_shape in [
            (4, 3, 1000),
            (10000,),
        ]:
            shpstr = "x".join([f"{x}" for x in data_shape])
            for dt, dtstr in [
                (np.dtype(np.int32), "i32"),
                (np.dtype(np.int64), "i64"),
            ]:
                is_int64 = dtstr == "i64"
                stream_len = data_shape[-1]
                leading_shape = data_shape[:-1]
                # Run identical tests on all processes (no MPI).
                input, _ = create_fake_data(data_shape, dtype=dt, sigma=None, comm=None)

                n_half = 5
                first = data_shape[-1] // 2 - n_half
                last = data_shape[-1] // 2 + n_half

                (compressed, stream_starts, stream_nbytes) = encode_flac(
                    input, level, use_threads=True
                )

                output = decode_flac(
                    compressed,
                    stream_starts,
                    stream_nbytes,
                    stream_len,
                    is_int64=is_int64,
                    use_threads=True,
                )
                out_shape = leading_shape + (stream_len,)
                output = output.reshape(out_shape)

                if not np.array_equal(output, input):
                    print(f"input_{dtstr}_{shpstr} = {input}", flush=True)
                    print(f"output_{dtstr}_{shpstr} = {output}", flush=True)
                    print(f"FAIL on {dtstr} roundtrip", flush=True)
                    self.assertTrue(False)

                # Decompress a slice and compare

                output = decode_flac(
                    compressed,
                    stream_starts,
                    stream_nbytes,
                    stream_len,
                    first_sample=first,
                    last_sample=last,
                    is_int64=is_int64,
                    use_threads=True,
                )
                out_shape = leading_shape + (last-first,)
                output = output.reshape(out_shape)

                slc = [slice(0, x) for x in input.shape]
                slc[-1] = slice(first, last)
                slc = tuple(slc)

                if not np.array_equal(output, input[slc]):
                    print(f"output_{dtstr}_{shpstr} = {output}", flush=True)
                    print(f"input_{dtstr}_{shpstr} = {input[slc]}", flush=True)
                    print(f"FAIL on {dtstr} roundtrip slice {slc}", flush=True)
                    self.assertTrue(False)
