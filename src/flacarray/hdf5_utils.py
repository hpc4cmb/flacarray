# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import tempfile

import numpy as np

try:
    import h5py
    have_hdf5 = True
except ImportError:
    have_hdf5 = False

from .mpi import MPI, use_mpi
from .utils import log


# Test whether h5py supports parallel I/O

_hdf5_is_parallel = None


def have_hdf5_parallel():
    global _hdf5_is_parallel
    if _hdf5_is_parallel is not None:
        # Already checked
        return _hdf5_is_parallel

    # Do we even have MPI?
    if not use_mpi:
        _hdf5_is_parallel = False
        return _hdf5_is_parallel

    # Try to open a temp file on each process with the mpio driver but using
    # COMM_SELF.  This lets us test the presence of the driver without actually
    # doing any communication
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            tfile = os.path.join(tempdir, f"test_hdf5_mpio_{MPI.COMM_WORLD.rank}.h5")
            with h5py.File(tfile, "w", driver="mpio", comm=MPI.COMM_SELF) as f:
                # Yay!
                _hdf5_is_parallel = True
    except (ValueError, AssertionError, AttributeError) as e:
        # Nope...
        _hdf5_is_parallel = False
    return _hdf5_is_parallel



def hdf5_use_serial(hgrp, mpi_comm):
    """Check if all processes in a communicator have access to the group.

    Args:
        hgrp (h5py.Group):  The local handle to the group.
        mpi_comm (MPI.Comm):  The MPI communicator or None.

    Returns:
        (bool):  True if the group is not open on all processes.

    """
    if mpi_comm is None:
        return True
    if mpi_comm.size == 1:
        return True
    # Have to check...
    total = mpi_comm.allreduce((1 if hgrp is not None else 0), op=MPI.SUM)
    if total != mpi_comm.size:
        return True
    else:
        return False


def hdf5_open(path, mode, comm=None, force_serial=False):
    """Open a file for reading or writing.

    This attempts to open the file with the mpio driver if available.  If
    not available or force_serial is True, then the file is opened on
    the rank zero process.

    Args:
        path (str):  The file path.
        mode (str):  The opening mode ("r", "w", etc).
        comm (MPI.Comm):  Optional MPI communicator.
        force_serial (bool):  If True, use serial HDF5 even if MPI is
            available.

    Returns:
        (h5py.File):  The opened file handle, or None if this process is
            not participating in I/O.

    """
    parallel = have_hdf5_parallel()
    if force_serial:
        parallel = False
    rank = 0
    if comm is not None:
        rank = comm.rank
    participating = parallel or (rank == 0)
    hf = None
    if participating:
        if parallel:
            hf = h5py.File(path, mode, driver="mpio", comm=comm)
            if rank == 0:
                log.debug(f"Opened file {path} in parallel")
        else:
            hf = h5py.File(path, mode)
            log.debug(f"Opened file {path} serially")
    return hf


class H5File(object):
    """Wrapper class containing an open HDF5 file.

    If the file is opened in serial mode with an MPI communicator, then
    The open file handle will be None on processes other than rank 0.

    """

    def __init__(self, name, mode, comm=None, force_serial=False):
        self.handle = hdf5_open(name, mode, comm=comm, force_serial=force_serial)

    def close(self):
        if hasattr(self, "handle") and self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def check_dataset_buffer_size(msg, slices, dtype, parallel):
    """Check the buffer size that will be used for I/O.

    When using HDF5 parallel I/O, reading or writing to a dataset with
    a buffer size > 2GB will cause an error.  This function checks the
    buffer size and issues a warning to provide more user feedback.

    Args:
        msg (str):  Message to write
        slices (tuple):  The slices that will be used for I/O
        dtype (numpy.dtype):  The data type
        parallel (bool):  Whether parallel h5py is enabled.

    Returns:
        None

    """
    if not parallel:
        # No issues
        return
    nelem = 1
    for slc in slices:
        nelem *= slc.stop - slc.start
    nbytes = nelem * dtype.itemsize
    if nbytes >= 2147483647:
        wmsg = f"{msg}:  buffer size of {nbytes} bytes > 2^31 - 1. "
        wmsg += "  HDF5 parallel I/O will likely fail."
        log.warning(wmsg)

