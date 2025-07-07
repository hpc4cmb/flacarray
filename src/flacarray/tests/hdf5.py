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

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        for local_shape in [(4, 3, 1000), (10000,)]:
            shpstr = "x".join([f"{int(x)}" for x in local_shape])
            for dt, dtstr, sigma, quant in [
                (np.dtype(np.int32), "i32", None, None),
                (np.dtype(np.int64), "i64", None, None),
                (np.dtype(np.float32), "f32", 1.0, 1.0e-7),
                (np.dtype(np.float64), "f64", 1.0, 1.0e-15),
            ]:
                input, mpi_dist = create_fake_data(
                    local_shape, sigma=sigma, dtype=dt, comm=self.comm
                )
                check = None
                filename = os.path.join(tmppath, f"data_{dtstr}_{shpstr}.h5")
                with H5File(filename, "w", comm=self.comm) as hf:
                    write_array(
                        input,
                        hf.handle,
                        level=5,
                        quanta=quant,
                        precision=None,
                        mpi_comm=self.comm,
                        use_threads=True,
                    )
                if self.comm is not None:
                    self.comm.barrier()
                with H5File(filename, "r", comm=self.comm) as hf:
                    check = read_array(
                        hf.handle,
                        keep=None,
                        stream_slice=None,
                        keep_indices=False,
                        mpi_comm=self.comm,
                        mpi_dist=mpi_dist,
                        use_threads=True,
                    )
                if dtstr == "i32" or dtstr == "i64":
                    local_fail = not np.array_equal(check, input)
                else:
                    local_fail = not np.allclose(check, input, atol=1e-6)
                if self.comm is not None:
                    fail = self.comm.allreduce(local_fail, op=MPI.SUM)
                else:
                    fail = local_fail
                if fail:
                    print(f"check_{dtstr}_{shpstr}[{rank}] = {check}", flush=True)
                    print(f"input_{dtstr}_{shpstr}[{rank}] = {input}", flush=True)
                    print(f"FAIL on {dtstr} roundtrip to hdf5", flush=True)
                    self.assertTrue(False)
        if self.comm is not None:
            self.comm.barrier()
        if tmpdir is not None:
            tmpdir.cleanup()
            del tmpdir

    def test_array_write_read(self):
        if not have_hdf5:
            print("h5py not available, skipping tests", flush=True)
            return
        if self.comm is None:
            rank = 0
        else:
            rank = self.comm.rank

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        for local_shape in [(4, 3, 1000), (10000,)]:
            shpstr = "x".join([f"{int(x)}" for x in local_shape])
            for dt, dtstr, sigma, quant in [
                (np.dtype(np.int32), "i32", None, None),
                (np.dtype(np.int64), "i64", None, None),
                (np.dtype(np.float32), "f32", 1.0, 1.0e-7),
                (np.dtype(np.float64), "f64", 1.0, 1.0e-15),
            ]:
                input, mpi_dist = create_fake_data(
                    local_shape, sigma=sigma, dtype=dt, comm=self.comm
                )
                flcarr = FlacArray.from_array(
                    input, quanta=quant, mpi_comm=self.comm, use_threads=True
                )

                filename = os.path.join(tmppath, f"data_{dtstr}_{shpstr}.h5")
                with H5File(filename, "w", comm=self.comm) as hf:
                    flcarr.write_hdf5(hf.handle)
                if self.comm is not None:
                    self.comm.barrier()
                with H5File(filename, "r", comm=self.comm) as hf:
                    check = FlacArray.read_hdf5(
                        hf.handle, mpi_comm=self.comm, mpi_dist=mpi_dist
                    )

                local_fail = check != flcarr
                if self.comm is not None:
                    fail = self.comm.allreduce(local_fail, op=MPI.SUM)
                else:
                    fail = local_fail

                if fail:
                    print(f"check_{dtstr}_{shpstr}[{rank}] = {check}", flush=True)
                    print(f"flcarr_{dtstr}_{shpstr}[{rank}] = {flcarr}", flush=True)
                    print(f"FAIL on {dtstr} FlacArray roundtrip to hdf5", flush=True)
                    self.assertTrue(False)
                else:
                    output = check.to_array(use_threads=True)
                    if dtstr == "i32" or dtstr == "i64":
                        local_arr_fail = not np.array_equal(output, input)
                    else:
                        local_arr_fail = not np.allclose(output, input, atol=1e-6)
                    if self.comm is not None:
                        arr_fail = self.comm.allreduce(local_arr_fail, op=MPI.SUM)
                    else:
                        arr_fail = local_arr_fail
                    if arr_fail:
                        print(f"output_{dtstr}_{shpstr}[{rank}] = {output}", flush=True)
                        print(f"input_{dtstr}_{shpstr}[{rank}] = {input}", flush=True)
                        print(f"FAIL on {dtstr} array roundtrip to hdf5", flush=True)
                        self.assertTrue(False)

        if self.comm is not None:
            self.comm.barrier()
        if tmpdir is not None:
            tmpdir.cleanup()
            del tmpdir
