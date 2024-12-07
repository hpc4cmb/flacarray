# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import unittest

import numpy as np

from ..libflacarray import (
    wrap_encode,
    wrap_encode_threaded,
    wrap_decode,
    encode_flac,
    decode_flac,
)


class BindingsTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]

    def test_wrappers(self):
        n_streams = 3
        stream_len = 10000
        level = 5

        flatsize = n_streams * stream_len

        rng = np.random.default_rng()
        data = rng.integers(low=-2 * 28, high=2 * 28, size=flatsize, dtype=np.int32)

        compressed, stream_starts, stream_nbytes = wrap_encode_threaded(
            data, n_streams, stream_len, level
        )

        output = wrap_decode(
            compressed, stream_starts, stream_nbytes, n_streams, stream_len, -1, -1, True
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

        output_slc = wrap_decode(
            compressed, stream_starts, stream_nbytes, n_streams, stream_len, first, last, True
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
        n_streams = 3
        stream_len = 10000
        level = 5

        flatsize = n_streams * stream_len

        rng = np.random.default_rng()
        data = rng.integers(
            low=-(2**28), high=2**28, size=flatsize, dtype=np.int32
        ).reshape((n_streams, stream_len))

        (compressed, stream_starts, stream_nbytes) = encode_flac(data, level, use_threads=True)

        output = decode_flac(compressed, stream_starts, stream_nbytes, stream_len, use_threads=True)

        for istream in range(n_streams):
            for isamp in range(stream_len):
                if output[istream, isamp] != data[istream, isamp]:
                    msg = f"FAIL [{istream},{isamp}]: {output[istream, isamp]} "
                    msg += f"!= {data[istream, isamp]}"
                    print(msg)
                    self.assertTrue(False)
