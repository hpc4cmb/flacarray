# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import tempfile
import unittest

import numpy as np

from ..array import FlacArray
from ..demo import create_fake_data
from ..hdf5 import write_array, read_array
from ..hdf5_utils import H5File, have_hdf5
from ..mpi import use_mpi, MPI

if have_hdf5:
    import h5py


class HDF5Test(unittest.TestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        if use_mpi:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = None

    def test_direct_write_read(self):
        if not have_hdf5:
            print("h5py not available, skipping tests", flush=True)
            return
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank

        data_shape = (4, 3, 1000)
        flatsize = np.prod(data_shape)
        rng = np.random.default_rng()

        input32 = rng.integers(
            low=-(2**29), high=2**29, size=flatsize, dtype=np.int32
        ).reshape(data_shape)
        check_i32 = None

        input64 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(data_shape)
            .astype(np.int64)
        )
        check_i64 = None

        inputf32 = create_fake_data(data_shape, 1.0).astype(np.float32)
        check_f32 = None

        inputf64 = create_fake_data(data_shape, 1.0)
        check_f64 = None

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        i32_file = os.path.join(tmppath, "data_i32.h5")
        with H5File(i32_file, "w", comm=self.comm) as hf:
            write_array(
                input32,
                hf.handle,
                level=5,
                quanta=None,
                precision=None,
                mpi_comm=self.comm,
                use_threads=True,
            )
        if self.comm is not None:
            self.comm.barrier()
        with H5File(i32_file, "r", comm=self.comm) as hf:
            check_i32 = read_array(
                hf.handle,
                keep=None,
                stream_slice=None,
                keep_indices=False,
                mpi_comm=self.comm,
                mpi_dist=None,
                use_threads=True,
            )

        i64_file = os.path.join(tmppath, "data_i64.h5")
        with H5File(i64_file, "w", comm=self.comm) as hf:
            write_array(
                input64,
                hf.handle,
                level=5,
                quanta=None,
                precision=None,
                mpi_comm=self.comm,
                use_threads=True,
            )
        if self.comm is not None:
            self.comm.barrier()
        with H5File(i64_file, "r", comm=self.comm) as hf:
            check_i64 = read_array(
                hf.handle,
                keep=None,
                stream_slice=None,
                keep_indices=False,
                mpi_comm=self.comm,
                mpi_dist=None,
                use_threads=True,
            )

        f32_file = os.path.join(tmppath, "data_f32.h5")
        with H5File(f32_file, "w", comm=self.comm) as hf:
            write_array(
                inputf32,
                hf.handle,
                level=5,
                quanta=None,
                precision=None,
                mpi_comm=self.comm,
                use_threads=True,
            )
        if self.comm is not None:
            self.comm.barrier()
        with H5File(f32_file, "r", comm=self.comm) as hf:
            check_f32 = read_array(
                hf.handle,
                keep=None,
                stream_slice=None,
                keep_indices=False,
                mpi_comm=self.comm,
                mpi_dist=None,
                use_threads=True,
            )

        f64_file = os.path.join(tmppath, "data_f64.h5")
        with H5File(f64_file, "w", comm=self.comm) as hf:
            write_array(
                inputf64,
                hf.handle,
                level=5,
                quanta=None,
                precision=None,
                mpi_comm=self.comm,
                use_threads=True,
            )
        if self.comm is not None:
            self.comm.barrier()
        with H5File(f64_file, "r", comm=self.comm) as hf:
            check_f64 = read_array(
                hf.handle,
                keep=None,
                stream_slice=None,
                keep_indices=False,
                mpi_comm=self.comm,
                mpi_dist=None,
                use_threads=True,
            )

        del tmppath
        del tmpdir

        if not np.array_equal(check_i32, input32):
            print(f"check_i32 = {check_i32}", flush=True)
            print(f"input_i32 = {input32}", flush=True)
            print("FAIL on i32 roundtrip to hdf5", flush=True)
            self.assertTrue(False)

        if not np.array_equal(check_i64, input64):
            print(f"check_i64 = {check_i64}", flush=True)
            print(f"input_i64 = {input64}", flush=True)
            print("FAIL on i64 roundtrip to hdf5", flush=True)
            self.assertTrue(False)

        if not np.allclose(check_f32, inputf32, atol=1e-6):
            print(f"check_f32 = {check_f32}", flush=True)
            print(f"input_f32 = {inputf32}", flush=True)
            print("FAIL on f32 roundtrip to hdf5", flush=True)
            self.assertTrue(False)

        if not np.allclose(check_f64, inputf64, atol=1e-6):
            print(f"check_f64 = {check_f64}", flush=True)
            print(f"input_f64 = {inputf64}", flush=True)
            print("FAIL on f64 roundtrip to hdf5", flush=True)
            self.assertTrue(False)

    def test_array_write_read(self):
        if not have_hdf5:
            print("h5py not available, skipping tests", flush=True)
            return
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank

        data_shape = (4, 3, 1000)
        flatsize = np.prod(data_shape)
        rng = np.random.default_rng()

        input32 = rng.integers(
            low=-(2**29), high=2**29, size=flatsize, dtype=np.int32
        ).reshape(data_shape)
        flcarr_i32 = FlacArray.from_array(input32, mpi_comm=self.comm, use_threads=True)
        check_i32 = None

        input64 = (
            rng.integers(low=-(2**27), high=2**30, size=flatsize, dtype=np.int32)
            .reshape(data_shape)
            .astype(np.int64)
        )
        flcarr_i64 = FlacArray.from_array(input64, mpi_comm=self.comm, use_threads=True)
        check_i64 = None

        inputf32 = create_fake_data(data_shape, 1.0).astype(np.float32)
        flcarr_f32 = FlacArray.from_array(
            inputf32, mpi_comm=self.comm, use_threads=True
        )
        check_f32 = None

        inputf64 = create_fake_data(data_shape, 1.0)
        flcarr_f64 = FlacArray.from_array(
            inputf64, mpi_comm=self.comm, use_threads=True
        )
        check_f64 = None

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        i32_file = os.path.join(tmppath, "data_i32.h5")
        with H5File(i32_file, "w", comm=self.comm) as hf:
            flcarr_i32.write_hdf5(hf.handle)
        if self.comm is not None:
            self.comm.barrier()
        with H5File(i32_file, "r", comm=self.comm) as hf:
            check_i32 = FlacArray.read_hdf5(hf.handle, mpi_comm=self.comm)

        i64_file = os.path.join(tmppath, "data_i64.h5")
        with H5File(i64_file, "w", comm=self.comm) as hf:
            flcarr_i64.write_hdf5(hf.handle)
        if self.comm is not None:
            self.comm.barrier()
        with H5File(i64_file, "r", comm=self.comm) as hf:
            check_i64 = FlacArray.read_hdf5(hf.handle, mpi_comm=self.comm)

        f32_file = os.path.join(tmppath, "data_f32.h5")
        with H5File(f32_file, "w", comm=self.comm) as hf:
            flcarr_f32.write_hdf5(hf.handle)
        if self.comm is not None:
            self.comm.barrier()
        with H5File(f32_file, "r", comm=self.comm) as hf:
            check_f32 = FlacArray.read_hdf5(hf.handle, mpi_comm=self.comm)

        f64_file = os.path.join(tmppath, "data_f64.h5")
        with H5File(f64_file, "w", comm=self.comm) as hf:
            flcarr_f64.write_hdf5(hf.handle)
        if self.comm is not None:
            self.comm.barrier()
        with H5File(f64_file, "r", comm=self.comm) as hf:
            check_f64 = FlacArray.read_hdf5(hf.handle, mpi_comm=self.comm)

        del tmppath
        del tmpdir

        if check_i32 != flcarr_i32:
            print(f"check_i32 = {check_i32}", flush=True)
            print(f"flcarr_i32 = {flcarr_i32}", flush=True)
            print("FAIL on i32 roundtrip to hdf5", flush=True)
            self.assertTrue(False)
        else:
            output_i32 = check_i32.to_array(use_threads=True)
            if not np.array_equal(output_i32, input32):
                print("FAIL on i32 hdf5 decompressed array check", flush=True)
                self.assertTrue(False)

        if check_i64 != flcarr_i64:
            print(f"check_i64 = {check_i64}", flush=True)
            print(f"flcarr_i64 = {flcarr_i64}", flush=True)
            print("FAIL on i64 roundtrip to hdf5", flush=True)
            self.assertTrue(False)
        else:
            output_i64 = check_i64.to_array(use_threads=True)
            if not np.array_equal(output_i64, input64):
                print("FAIL on i64 hdf5 decompressed array check", flush=True)
                self.assertTrue(False)

        if check_f32 != flcarr_f32:
            print("FAIL on f32 roundtrip to hdf5", flush=True)
            self.assertTrue(False)
        else:
            output_f32 = check_f32.to_array(use_threads=True)
            if not np.allclose(output_f32, inputf32, rtol=1.0e-5, atol=1.0e-5):
                print("FAIL on f32 hdf5 decompressed array check", flush=True)
                self.assertTrue(False)

        if check_f64 != flcarr_f64:
            print("FAIL on f64 roundtrip to hdf5", flush=True)
            self.assertTrue(False)
        else:
            output_f64 = check_f64.to_array(use_threads=True)
            if not np.allclose(output_f64, inputf64, rtol=1.0e-5, atol=1.0e-5):
                print("FAIL on f64 hdf5 decompressed array check", flush=True)
                self.assertTrue(False)
