# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import argparse
import os
import time

import numpy as np
import zarr

from ..array import FlacArray
from ..demo import create_fake_data
from ..hdf5 import write_array as hdf5_write_array
from ..hdf5 import read_array as hdf5_read_array
from ..hdf5_utils import H5File
from ..mpi import use_mpi, MPI
from ..utils import print_timers
from ..zarr import write_array as zarr_write_array
from ..zarr import read_array as zarr_read_array


def benchmark(
    shape, dir=".", keep=None, stream_slice=None, mpi_comm=None, use_threads=False
):
    """Run benchmarks.

    This will create some fake data with the specified shape and then test different
    writing and reading patterns.

    """
    rank = 0
    if mpi_comm is not None:
        rank = mpi_comm.rank

    if rank == 0:
        os.makedirs(dir, exist_ok=True)
    if mpi_comm is not None:
        mpi_comm.barrier()

    arr = create_fake_data(shape)
    shpstr = "-".join([f"{x}" for x in shape])
    print(f"  Running tests with shape {shape} ({arr.nbytes} bytes)", flush=True)

    # Run HDF5 tests

    start = time.perf_counter()
    flcarr = FlacArray.from_array(arr, use_threads=use_threads)
    stop = time.perf_counter()
    if mpi_comm is not None:
        mpi_comm.barrier()
    if rank == 0:
        print(f"  FlacArray compress in {stop-start:0.3f} seconds", flush=True)

    out_file = os.path.join(dir, f"io_bench_{shpstr}.h5")
    with H5File(out_file, "w", comm=mpi_comm) as hf:
        start = time.perf_counter()
        flcarr.write_hdf5(hf.handle)
        stop = time.perf_counter()
        if rank == 0:
            print(f"  FlacArray write HDF5 in {stop-start:0.3f} seconds", flush=True)

    check = None
    with H5File(out_file, "r", comm=mpi_comm) as hf:
        start = time.perf_counter()
        check = FlacArray.read_hdf5(hf.handle, keep=keep, mpi_comm=mpi_comm)
        stop = time.perf_counter()
        if rank == 0:
            print(f"  FlacArray read HDF5 in {stop-start:0.3f} seconds", flush=True)

    del flcarr
    del check

    # Run Zarr tests

    flcarr = FlacArray.from_array(arr, use_threads=use_threads)

    out_file = os.path.join(dir, f"io_bench_{shpstr}.zarr")
    with zarr.open_group(out_file, mode="w") as zf:
        start = time.perf_counter()
        flcarr.write_zarr(zf)
        stop = time.perf_counter()
        if rank == 0:
            print(f"  FlacArray write Zarr in {stop-start:0.3f} seconds", flush=True)

    check = None
    with zarr.open_group(out_file, mode="r") as zf:
        start = time.perf_counter()
        check = FlacArray.read_zarr(zf, keep=keep, mpi_comm=mpi_comm)
        stop = time.perf_counter()
        if rank == 0:
            print(f"  FlacArray read Zarr in {stop-start:0.3f} seconds", flush=True)

    del flcarr
    del check

    # Direct I/O

    # HDF5

    out_file = os.path.join(dir, f"io_bench_direct_{shpstr}.h5")
    with H5File(out_file, "w", comm=mpi_comm) as hf:
        start = time.perf_counter()
        hdf5_write_array(
            arr, hf.handle, level=5, mpi_comm=mpi_comm, use_threads=use_threads
        )
        stop = time.perf_counter()
        if rank == 0:
            print(
                f"  Direct compress and write HDF5 in {stop-start:0.3f} seconds",
                flush=True,
            )

    check = None
    with H5File(out_file, "r", comm=mpi_comm) as hf:
        start = time.perf_counter()
        check = hdf5_read_array(
            hf.handle,
            keep=keep,
            stream_slice=stream_slice,
            mpi_comm=mpi_comm,
            use_threads=use_threads,
        )
        stop = time.perf_counter()
        if rank == 0:
            print(
                f"  Direct read HDF5 and decompress in {stop-start:0.3f} seconds",
                flush=True,
            )

    # Zarr

    out_file = os.path.join(dir, f"io_bench_direct_{shpstr}.zarr")
    with zarr.open_group(out_file, mode="w") as zf:
        start = time.perf_counter()
        zarr_write_array(arr, zf, level=5, mpi_comm=mpi_comm, use_threads=use_threads)
        stop = time.perf_counter()
        if rank == 0:
            print(
                f"  Direct compress and write Zarr in {stop-start:0.3f} seconds",
                flush=True,
            )

    check = None
    with zarr.open_group(out_file, mode="r") as zf:
        start = time.perf_counter()
        check = zarr_read_array(
            zf,
            keep=keep,
            stream_slice=stream_slice,
            mpi_comm=mpi_comm,
            use_threads=use_threads,
        )
        stop = time.perf_counter()
        if rank == 0:
            print(
                f"  Direct read Zarr and decompress in {stop-start:0.3f} seconds",
                flush=True,
            )

    del check
    del arr

    if rank == 0:
        print_timers()


def cli():
    parser = argparse.ArgumentParser(description="Run Benchmarks")
    parser.add_argument(
        "--out_dir",
        required=False,
        default="flacarray_benchmark_out",
        help="Output directory",
    )
    parser.add_argument(
        "--data_shape",
        required=False,
        default="(4,3,100000)",
        help="Data shape (as a string)",
    )
    parser.add_argument(
        "--use_threads",
        required=False,
        default=False,
        action="store_true",
        help="Use OpenMP threads",
    )
    args = parser.parse_args()

    shape = eval(args.data_shape)

    if use_mpi:
        comm = MPI.COMM_WORLD
    else:
        comm = None

    print("Full Data Tests:", flush=True)
    out = os.path.join(args.out_dir, "full")
    benchmark(shape, dir=out, use_threads=args.use_threads, mpi_comm=comm)

    # Now try with a keep mask and sample slice
    keep = np.zeros(shape[:-1], dtype=bool)
    for row in range(shape[0]):
        if row % 2 == 0:
            keep[row] = True
    mid = shape[-1] // 2
    samp_slice = slice(mid - 50, mid + 50, 1)

    print("Sliced Data Tests (100 samples from even stream indices):", flush=True)
    out = os.path.join(args.out_dir, "sliced")
    benchmark(
        shape,
        dir=out,
        keep=keep,
        stream_slice=samp_slice,
        use_threads=args.use_threads,
        mpi_comm=comm,
    )


if __name__ == "__main__":
    cli()
