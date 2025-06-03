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
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        local_shape = (4, 3, 1000)

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
            filename = os.path.join(tmppath, f"data_{dtstr}.zarr")
            with ZarrGroup(filename, mode="w", comm=self.comm) as zf:
                write_array(
                    input,
                    zf,
                    level=5,
                    quanta=quant,
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
                local_fail = not np.array_equal(check, input)
            else:
                local_fail = not np.allclose(check, input, atol=1e-6)
            if self.comm is not None:
                fail = self.comm.allreduce(local_fail, op=MPI.SUM)
            else:
                fail = local_fail
            if fail:
                print(f"check_{dtstr} = {check}", flush=True)
                print(f"input_{dtstr} = {input}", flush=True)
                print(f"FAIL on {dtstr} roundtrip to zarr", flush=True)
                self.assertTrue(False)
        if self.comm is not None:
            self.comm.barrier()
        if tmpdir is not None:
            tmpdir.cleanup()
            del tmpdir

    def test_array_write_read(self):
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
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        local_shape = (4, 3, 1000)

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

            filename = os.path.join(tmppath, f"data_{dtstr}.zarr")
            with ZarrGroup(filename, mode="w", comm=self.comm) as zf:
                flcarr.write_zarr(zf)
            if self.comm is not None:
                self.comm.barrier()
            with ZarrGroup(filename, mode="r", comm=self.comm) as zf:
                check = FlacArray.read_zarr(zf, mpi_comm=self.comm)

            # Check array equality
            local_fail = int(check != flcarr)
            if self.comm is not None:
                fail = self.comm.allreduce(local_fail, op=MPI.SUM)
            else:
                fail = local_fail

            if fail:
                print(f"check_{dtstr} = {check}", flush=True)
                print(f"flcarr_{dtstr} = {flcarr}", flush=True)
                print(f"FAIL on {dtstr} FlacArray roundtrip to zarr", flush=True)
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
                    print(f"output_{dtstr} = {output}", flush=True)
                    print(f"input_{dtstr} = {input}", flush=True)
                    print(f"FAIL on {dtstr} array roundtrip to zarr", flush=True)
                    self.assertTrue(False)
        if self.comm is not None:
            self.comm.barrier()
        if tmpdir is not None:
            tmpdir.cleanup()
            del tmpdir
