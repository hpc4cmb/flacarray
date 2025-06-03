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
from ..mpi import use_mpi, MPI, distribute_and_verify
from ..utils import print_timers
from ..zarr import write_array as zarr_write_array
from ..zarr import read_array as zarr_read_array
from ..zarr import ZarrGroup


def dump_debug_text(arr, dir):
    """Write out a text version of the compressed array data.

    This is only intended for debugging on small arrays!

    Args:
        arr (FlacArray):  The array to dump.
        dir (str):  The output directory.

    Returns:
        None

    """
    comm = arr.mpi_comm
    dist = arr.mpi_dist
    rank = 0
    if comm is not None:
        rank = comm.rank
    shpstr = "x".join([f"{x}" for x in arr.global_shape])
    dbgfile = os.path.join(dir, f"debug_bench_{shpstr}_{rank}.txt")

    first_leading = dist[rank][0]
    it = np.nditer(arr.stream_starts, order="C", flags=["multi_index"])

    with open(dbgfile, "w") as fd:
        for st in it:
            idx = it.multi_index
            start = arr.stream_starts[idx]
            nbyte = arr.stream_nbytes[idx]
            off = None
            if arr.stream_offsets is not None:
                off = arr.stream_offsets[idx]
            gain = None
            if arr.stream_gains is not None:
                gain = arr.stream_gains[idx]
            global_start = arr.global_stream_starts[idx]
            global_idx = [int(x) for x in idx]
            global_idx[0] += int(first_leading)
            global_idx = tuple(global_idx)
            fd.write(
                f"---- stream {rank} {first_leading} {idx} {global_idx}: {global_start}, {nbyte}, {off}, {gain}\n"
            )
            cdata = arr.compressed[start : start + nbyte]
            ncol = 30
            cur = 0
            cl = 0
            row = ""
            while cur < nbyte:
                if cl > ncol:
                    fd.write(f"{row}\n")
                    row = ""
                    cl = 0
                row = f"{row} {cdata[cur]:03d}"
                cl += 1
                cur += 1
            fd.write(f"{row}\n")
    it.close()
    del it

    outfile = os.path.join(dir, f"debug_bench_{shpstr}.txt")
    if comm is not None:
        comm.barrier()
        # Consolidate outputs
        all_files = comm.gather(dbgfile, root=0)
        if rank == 0:
            with open(outfile, "w") as fout:
                for fname in all_files:
                    with open(fname, "r") as fin:
                        fout.write(fin.read())
                    os.remove(fname)
    else:
        os.rename(dbgfile, outfile)


def benchmark(
    global_shape,
    dir=".",
    keep=None,
    stream_slice=None,
    mpi_comm=None,
    use_threads=False,
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

    # Get the local shape
    dist = distribute_and_verify(mpi_comm, global_shape[0])
    local_shape = [dist[rank][1] - dist[rank][0]]
    local_shape.extend(global_shape[1:])
    local_shape = tuple(local_shape)

    arr, mpi_dist = create_fake_data(local_shape, comm=mpi_comm)
    shpstr = "x".join([f"{x}" for x in global_shape])

    # Run HDF5 tests

    start = time.perf_counter()
    flcarr = FlacArray.from_array(
        arr, quanta=1.0e-15, mpi_comm=mpi_comm, use_threads=use_threads
    )
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(f"  FlacArray compress in {stop - start:0.3f} seconds", flush=True)

    # dump_debug_text(flcarr, dir)

    out_file = os.path.join(dir, f"io_bench_{shpstr}.h5")
    start = time.perf_counter()
    with H5File(out_file, "w", comm=mpi_comm) as hf:
        flcarr.write_hdf5(hf.handle)
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(f"  FlacArray write HDF5 in {stop - start:0.3f} seconds", flush=True)

    check = None
    start = time.perf_counter()
    with H5File(out_file, "r", comm=mpi_comm) as hf:
        check = FlacArray.read_hdf5(
            hf.handle, keep=keep, mpi_comm=mpi_comm, mpi_dist=mpi_dist
        )
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(f"  FlacArray read HDF5 in {stop - start:0.3f} seconds", flush=True)

    del flcarr
    del check

    return

    # Run Zarr tests

    flcarr = FlacArray.from_array(
        arr, quanta=1.0e-15, mpi_comm=mpi_comm, use_threads=use_threads
    )

    out_file = os.path.join(dir, f"io_bench_{shpstr}.zarr")
    start = time.perf_counter()
    with ZarrGroup(out_file, mode="w", comm=mpi_comm) as zf:
        flcarr.write_zarr(zf)
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(f"  FlacArray write Zarr in {stop - start:0.3f} seconds", flush=True)

    check = None
    start = time.perf_counter()
    with ZarrGroup(out_file, mode="r", comm=mpi_comm) as zf:
        check = FlacArray.read_zarr(zf, keep=keep, mpi_comm=mpi_comm, mpi_dist=mpi_dist)
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(f"  FlacArray read Zarr in {stop - start:0.3f} seconds", flush=True)

    del flcarr
    del check

    # Direct I/O

    # HDF5

    out_file = os.path.join(dir, f"io_bench_direct_{shpstr}.h5")
    start = time.perf_counter()
    with H5File(out_file, "w", comm=mpi_comm) as hf:
        hdf5_write_array(
            arr,
            hf.handle,
            level=5,
            quanta=1.0e-15,
            mpi_comm=mpi_comm,
            use_threads=use_threads,
            mpi_dist=mpi_dist,
        )
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(
            f"  Direct compress and write HDF5 in {stop - start:0.3f} seconds",
            flush=True,
        )

    check = None
    start = time.perf_counter()
    with H5File(out_file, "r", comm=mpi_comm) as hf:
        check = hdf5_read_array(
            hf.handle,
            keep=keep,
            stream_slice=stream_slice,
            mpi_comm=mpi_comm,
            use_threads=use_threads,
            mpi_dist=mpi_dist,
        )
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(
            f"  Direct read HDF5 and decompress in {stop - start:0.3f} seconds",
            flush=True,
        )

    # Zarr

    out_file = os.path.join(dir, f"io_bench_direct_{shpstr}.zarr")
    start = time.perf_counter()
    with ZarrGroup(out_file, mode="w", comm=mpi_comm) as zf:
        zarr_write_array(
            arr,
            zf,
            level=5,
            quanta=1.0e-15,
            mpi_comm=mpi_comm,
            use_threads=use_threads,
            mpi_dist=mpi_dist,
        )
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(
            f"  Direct compress and write Zarr in {stop - start:0.3f} seconds",
            flush=True,
        )

    check = None
    start = time.perf_counter()
    with ZarrGroup(out_file, mode="r", comm=mpi_comm) as zf:
        check = zarr_read_array(
            zf,
            keep=keep,
            stream_slice=stream_slice,
            mpi_comm=mpi_comm,
            use_threads=use_threads,
            mpi_dist=mpi_dist,
        )
    if mpi_comm is not None:
        mpi_comm.barrier()
    stop = time.perf_counter()
    if rank == 0:
        print(
            f"  Direct read Zarr and decompress in {stop - start:0.3f} seconds",
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
        "--global_shape",
        required=False,
        default="(4,3,100000)",
        help="Global data shape (as a string)",
    )
    parser.add_argument(
        "--use_threads",
        required=False,
        default=False,
        action="store_true",
        help="Use OpenMP threads",
    )
    args = parser.parse_args()

    shape = eval(args.global_shape)

    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        comm = None
        rank = 0

    if rank == 0:
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

    if rank == 0:
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
