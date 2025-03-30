# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

from .utils import log


if "MPI_DISABLE" in os.environ:
    use_mpi = False
    MPI = None
else:
    try:
        # related to MPI_THREAD_MULTIPLE and errors when OpenMPI 4 is used.
        # https://github.com/mpi4py/mpi4py/issues/34#issuecomment-800233017
        import mpi4py

        mpi4py.rc.thread_level = "serialized"

        import mpi4py.MPI as MPI

        use_mpi = True
    except Exception:
        # There could be many possible exceptions raised...
        log.debug("mpi4py not found- MPI operations disabled")
        use_mpi = False
        MPI = None


def distribute_and_verify(mpi_comm, n_elem, mpi_dist=None):
    """Compute or verify a distribution of elements across a communicator.

    If `mpi_dist` is specified, the contents are checked for consistency with
    the specified communicator and number of elements.  If `mpi_dist` is not
    specified, it is computed from the size of the communicator and distributing
    the elements uniformly across processes.

    Args:
        mpi_comm (MPI.Comm):  The MPI communicator (or None)
        n_elem (int):  The number of elements to distribute.
        mpi_dist (list):  If specified, the input will be verified and returned.

    Returns:
        (list):  The verified or created MPI distribution.

    """
    if mpi_dist is not None:
        if mpi_comm is None:
            # Not using MPI, so the dist better contain just the full range
            if len(mpi_dist) != 1 or mpi_dist[0][0] != 0 or mpi_dist[0][1] != n_elem:
                msg = "mpi_comm is None and mpi_dist does not contain single range "
                msg += "of all elements"
                raise RuntimeError(msg)
            return mpi_dist
        if mpi_comm.size != len(mpi_dist):
            msg = f"If specified, mpi_dist (len={len(mpi_dist)}) should have same "
            msg += f"length as comm size ({mpi_comm.size})"
            raise RuntimeError(msg)
        if mpi_dist[0][0] != 0 or mpi_dist[-1][1] != n_elem:
            msg = f"If specified, mpi_dist ({mpi_dist[0][0]} ... {mpi_dist[-1][1]})"
            msg += f" should span the full range of elements ({n_elem})"
            raise RuntimeError(msg)
        for proc in range(1, mpi_comm.size):
            if mpi_dist[proc][0] != mpi_dist[proc - 1][1]:
                raise RuntimeError(
                    "mpi_dist must have contiguous ranges of first, last (exclusive)"
                )
            if mpi_dist[proc][1] <= mpi_dist[proc][0]:
                raise RuntimeError(
                    f"mpi_dist has no data for process {proc}"
                )
        # Everything checks out
        return mpi_dist
    else:
        # We are creating the distribution
        if mpi_comm is None:
            # One range
            return [(0, n_elem)]
        else:
            # Compute the mpi_dist- just uniform distribution
            chunks = np.array_split(np.arange(n_elem, dtype=np.int64), mpi_comm.size)
            for proc, ch in enumerate(chunks):
                if len(ch) == 0:
                    msg = f"Cannot distribute {n_elem} streams among {mpi_comm.size}"
                    msg += " processes."
                    raise RuntimeError(msg)
            return [(x[0], x[-1] + 1) for x in chunks]


def global_array_properties(local_shape, mpi_comm):
    """Compute various properties of the global data distribution.

    Given the local data properties on each process and the MPI communicator,
    compute various useful quantities for working with global data.

    This function also verifies that non-leading dimensions match across all processes.

    Args:
        local_shape (tuple):  The local data shape on each process.
        mpi_comm (MPI.Comm):  The MPI communicator or None.

    Returns:
        (dict):  The dictionary of global properties.

    """
    props = dict()
    if mpi_comm is None:
        if len(local_shape) == 1:
            # Just one stream
            props["shape"] = (1, local_shape[0])
            props["dist"] = [(0, 1)]
        else:
            props["shape"] = local_shape
            props["dist"] = [(0, local_shape[0])]
        return props
    all_shapes = mpi_comm.gather(local_shape, root=0)
    err = False
    if mpi_comm.rank == 0:
        dist = list()
        shp = all_shapes[0]
        if len(shp) == 1:
            lda = 1
            trl = shp
        else:
            lda = shp[0]
            trl = shp[1:]
        dist.append((0, lda))
        ldoff = lda
        for s in all_shapes[1:]:
            if len(s) == 1:
                lda += 1
                dist.append((ldoff, ldoff + 1))
                ldoff += 1
                if s != trl:
                    err = True
                    break
            else:
                lda += s[0]
                dist.append((ldoff, ldoff + s[0]))
                ldoff += s[0]
                if s[1:] != trl:
                    err = True
                    break
        props["shape"] = (lda,) + trl
        props["dist"] = dist
    err = mpi_comm.bcast(err, root=0)
    props = mpi_comm.bcast(props, root=0)
    if err:
        raise RuntimeError("Inconsistent array dimensions across processes")
    return props


def global_bytes(local_nbytes, stream_starts, mpi_comm):
    """Compute properties of the global compressed bytestream.

    The number of bytes on each process is communicated to all processes.  From
    this information the global total bytes and the global starting bytes for streams
    on the local process are computed.  The array of global starts is the same shape
    as `local_starts`.

    Args:
        local_nbytes (int):  The total compressed bytes on this process.
        stream_starts (array):  Array of local byte offsets for each stream.
        mpi_comm (MPI.Comm):  The MPI communicator or None.

    Returns:
        (tuple):  The (total global bytes, bytes per process, global byte offsets).

    """
    if mpi_comm is None or mpi_comm.size == 1:
        return (local_nbytes, [local_nbytes], stream_starts)
    rank = mpi_comm.rank
    nproc = mpi_comm.size
    all_nbytes = mpi_comm.allgather(local_nbytes)
    global_nbytes = np.sum(all_nbytes)
    # Each process goes through the list of bytes-per-process and computes the
    # global offsets for its data.
    byte_offset = 0
    for iproc in range(nproc):
        if iproc == rank:
            global_starts = stream_starts + byte_offset
            break
        byte_offset += all_nbytes[iproc]
    return (global_nbytes, all_nbytes, global_starts)
