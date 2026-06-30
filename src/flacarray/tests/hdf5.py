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

    def test_array_write_read_nodata(self):
        if not have_hdf5:
            print("h5py not available, skipping tests", flush=True)
            return
        if self.comm is None or self.comm.size < 2:
            print("Less than 2 processes, skipping MPI test with empty procs")
            return

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
                if rank == self.comm.size - 1:
                    trailing = local_shape[1:]
                    if len(trailing) == 0:
                        input_shape = (0,)
                    else:
                        input_shape = (0,) + trailing
                else:
                    input_shape = local_shape
                input, mpi_dist = create_fake_data(
                    input_shape, sigma=sigma, dtype=dt, comm=self.comm
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

    def test_array_keep_dist(self):
        if not have_hdf5:
            print("h5py not available, skipping tests", flush=True)
            return
        if self.comm is None:
            rank = 0
            nproc = 1
        else:
            rank = self.comm.rank
            nproc = self.comm.size

        tmpdir = None
        tmppath = None
        if rank == 0:
            tmpdir = tempfile.TemporaryDirectory()
            tmppath = tmpdir.name
        if self.comm is not None:
            tmppath = self.comm.bcast(tmppath, root=0)

        n_local_stream = 10
        n_sample = 1000
        n_stream = n_local_stream * nproc

        local_shape = (n_local_stream, n_sample)
        shpstr = "x".join([f"{int(x)}" for x in local_shape])
        dt = np.dtype(np.int64)
        dtstr = "i64"

        input, _ = create_fake_data(local_shape, sigma=None, dtype=dt, comm=self.comm)

        # Each process needs to compare its loaded array slice with the input.
        # Since the input is small, we just get a copy on all ranks.
        if nproc > 1:
            input_chunks = self.comm.allgather(input)
            global_input = np.concatenate(input_chunks, axis=0)
        else:
            global_input = input

        filename = os.path.join(tmppath, f"data_{dtstr}_{shpstr}.h5")
        with H5File(filename, "w", comm=self.comm) as hf:
            write_array(input, hf.handle, level=5, mpi_comm=self.comm)
        if self.comm is not None:
            self.comm.barrier()

        # When reading the data, use an arbitrary keep array and MPI
        # distribution.

        keep_mod = 4
        keep = np.zeros(n_stream, dtype=bool)
        for istr in range(n_stream):
            if istr % keep_mod == 0 and istr != 0:
                keep[istr] = 1

        if nproc == 1:
            mpi_dist = [(0, n_stream)]
        else:
            # Give the first rank just a couple streams and then divide the rest.
            # We intentionally test the case where some ranks have no streams.
            n_first = keep_mod // 2
            n_remain = n_stream - n_first

            mpi_dist = [(0, n_first)]
            chunks = np.array_split(np.arange(n_remain, dtype=np.int32), nproc - 1)
            for proc, ch in enumerate(chunks):
                if len(ch) == 0:
                    msg = f"Cannot distribute {n_remain} streams among {nproc - 1}"
                    msg += " remaining processes."
                    raise RuntimeError(msg)
                mpi_dist.append((int(n_first + ch[0]), int(n_first + ch[-1] + 1)))

        # Get the expected streams we should have on each process
        local_slc = slice(mpi_dist[rank][0], mpi_dist[rank][1], 1)
        local_streams = np.arange(n_stream, dtype=np.int32)[local_slc]
        local_keep = keep[local_slc]

        with H5File(filename, "r", comm=self.comm) as hf:
            loaded = read_array(
                hf.handle,
                keep=keep,
                mpi_comm=self.comm,
                mpi_dist=mpi_dist,
                no_flatten=True,
            )

        # Check out local data
        local_fail = 0
        lstr = 0
        if loaded is not None:
            # This process has some data
            for gstr, lkp in zip(local_streams, local_keep):
                if not lkp:
                    continue
                if not np.array_equal(loaded[lstr], global_input[gstr]):
                    msg = f"P[{rank}] loaded global stream {gstr} to local {lstr}, "
                    msg += f"values disagree ({loaded[lstr][0]}, {loaded[lstr][1]}, ..."
                    msg += f" != {global_input[gstr][0]}, {global_input[gstr][1]}, ...)"
                    print(msg, flush=True)
                    local_fail = 1
                lstr += 1

        if self.comm is not None:
            fail = self.comm.allreduce(local_fail, op=MPI.SUM)
        else:
            fail = local_fail

        if fail > 0:
            if rank == 0:
                print("FAIL on mpi_dist load with keep array", flush=True)
            self.assertTrue(False)

        if self.comm is not None:
            self.comm.barrier()
        if tmpdir is not None:
            tmpdir.cleanup()
            del tmpdir
