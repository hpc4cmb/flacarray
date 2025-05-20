# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import tempfile
import unittest

import numpy as np

from ..array import FlacArray
from ..demo import create_fake_data
from ..zarr import have_zarr, write_array, read_array, ZarrGroup
from ..mpi import use_mpi, MPI

if have_zarr:
    import zarr


class ZarrTest(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if use_mpi:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = None

    def test_direct_write_read(self):
        if not have_zarr:
            print("zarr not available, skipping tests", flush=True)
            return
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory(delete=False)
            tmppath = tmpdir.name
            print(tmppath)
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        local_shape = (4, 3, 1000)

        for dt, dtstr, sigma in [
            (np.dtype(np.int32), "i32", None),
            (np.dtype(np.int64), "i64", None),
            (np.dtype(np.float32), "f32", 1.0),
            (np.dtype(np.float64), "f64", 1.0),
        ]:
            input, mpi_dist = create_fake_data(
                local_shape, sigma=sigma, dtype=dt, comm=self.comm
            )
            check = None
            filename = os.path.join(tmppath, f"data_{dtstr}.zarr")
            with ZarrGroup(filename, mode="w", comm=self.comm) as zf:
                write_array(
                    input,
                    zf,
                    level=5,
                    quanta=None,
                    precision=None,
                    mpi_comm=self.comm,
                    use_threads=True,
                )
            if self.comm is not None:
                self.comm.barrier()
            with ZarrGroup(filename, mode="r", comm=self.comm) as zf:
                check = read_array(
                    zf,
                    keep=None,
                    stream_slice=None,
                    keep_indices=False,
                    mpi_comm=self.comm,
                    mpi_dist=mpi_dist,
                    use_threads=True,
                )
            if dtstr == "i32" or dtstr == "i64":
                if not np.array_equal(check, input):
                    print(f"check_{dtstr} = {check}", flush=True)
                    print(f"input_{dtstr} = {input}", flush=True)
                    print("FAIL on {dtstr} roundtrip to zarr", flush=True)
                    self.assertTrue(False)
            else:
                if not np.allclose(check, input, atol=1e-6):
                    print(f"check_{dtstr} = {check}", flush=True)
                    print(f"input_{dtstr} = {input}", flush=True)
                    print(f"FAIL on {dtstr} roundtrip to zarr", flush=True)
                    self.assertTrue(False)

        if self.comm is not None:
            self.comm.barrier()
        del tmppath
        del tmpdir

    def test_array_write_read(self):
        if not have_zarr:
            print("zarr not available, skipping tests", flush=True)
            return
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank

        local_shape = (4, 3, 1000)
        flatsize = np.prod(local_shape)
        rng = np.random.default_rng()

        input32 = rng.integers(
            low=-(2**29), high=2**29, size=flatsize, dtype=np.int32
        ).reshape(local_shape)
        flcarr_i32 = FlacArray.from_array(input32, mpi_comm=self.comm, use_threads=True)
        check_i32 = None

        input64 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(local_shape)
            .astype(np.int64)
        )
        flcarr_i64 = FlacArray.from_array(input64, mpi_comm=self.comm, use_threads=True)
        check_i64 = None

        inputf32, _ = create_fake_data(local_shape, 1.0, comm=self.comm)
        inputf32 = inputf32.astype(np.float32)
        flcarr_f32 = FlacArray.from_array(inputf32, mpi_comm=self.comm)
        check_f32 = None

        inputf64, _ = create_fake_data(local_shape, 1.0, comm=self.comm)
        flcarr_f64 = FlacArray.from_array(inputf64, mpi_comm=self.comm, use_threads=True)
        check_f64 = None

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        i32_file = os.path.join(tmppath, "data_i32.zarr")
        with ZarrGroup(i32_file, mode="w", comm=self.comm) as zf:
            flcarr_i32.write_zarr(zf)
        if self.comm is not None:
            self.comm.barrier()
        with ZarrGroup(i32_file, mode="r", comm=self.comm) as zf:
            check_i32 = FlacArray.read_zarr(zf, mpi_comm=self.comm)

        i64_file = os.path.join(tmppath, "data_i64.zarr")
        with ZarrGroup(i64_file, mode="w", comm=self.comm) as zf:
            flcarr_i64.write_zarr(zf)
        if self.comm is not None:
            self.comm.barrier()
        with ZarrGroup(i64_file, mode="r", comm=self.comm) as zf:
            check_i64 = FlacArray.read_zarr(zf, mpi_comm=self.comm)

        f32_file = os.path.join(tmppath, "data_f32.zarr")
        with ZarrGroup(f32_file, mode="w", comm=self.comm) as zf:
            flcarr_f32.write_zarr(zf)
        if self.comm is not None:
            self.comm.barrier()
        with ZarrGroup(f32_file, mode="r", comm=self.comm) as zf:
            check_f32 = FlacArray.read_zarr(zf, mpi_comm=self.comm)

        f64_file = os.path.join(tmppath, "data_f64.zarr")
        with ZarrGroup(f64_file, mode="w", comm=self.comm) as zf:
            flcarr_f64.write_zarr(zf)
        if self.comm is not None:
            self.comm.barrier()
        with ZarrGroup(f64_file, mode="r", comm=self.comm) as zf:
            check_f64 = FlacArray.read_zarr(zf, mpi_comm=self.comm)

        if self.comm is not None:
            self.comm.barrier()
        del tmppath
        del tmpdir

        if check_i32 != flcarr_i32:
            print(f"check_i32 = {check_i32}", flush=True)
            print(f"flcarr_i32 = {flcarr_i32}", flush=True)
            print("FAIL on i32 roundtrip to zarr", flush=True)
            self.assertTrue(False)
        else:
            output_i32 = check_i32.to_array()
            if not np.array_equal(output_i32, input32):
                print("FAIL on i32 zarr decompressed array check", flush=True)
                self.assertTrue(False)

        if check_i64 != flcarr_i64:
            print("FAIL on i64 roundtrip to zarr", flush=True)
            self.assertTrue(False)
        else:
            output_i64 = check_i64.to_array()
            if not np.array_equal(output_i64, input64):
                print("FAIL on i64 zarr decompressed array check", flush=True)
                self.assertTrue(False)

        if check_f32 != flcarr_f32:
            print("FAIL on f32 roundtrip to zarr", flush=True)
            self.assertTrue(False)
        else:
            output_f32 = check_f32.to_array()
            if not np.allclose(output_f32, inputf32, rtol=1.0e-5, atol=1.0e-5):
                print("FAIL on f32 zarr decompressed array check", flush=True)
                self.assertTrue(False)

        if check_f64 != flcarr_f64:
            print("FAIL on f64 roundtrip to zarr", flush=True)
            self.assertTrue(False)
        else:
            output_f64 = check_f64.to_array()
            if not np.allclose(output_f64, inputf64, rtol=1.0e-5, atol=1.0e-5):
                print("FAIL on f64 zarr decompressed array check", flush=True)
                self.assertTrue(False)
