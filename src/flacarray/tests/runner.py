# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import sys
import unittest

from ..mpi import use_mpi, MPI

from . import array as test_array
from . import bindings as test_bindings
from . import hdf5 as test_hdf5
from . import utils as test_utils
from . import zarr as test_zarr


def test(name=None):
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestSuite()

    if name is not None:
        # Add just this specific test
        modname = f"flacarray.tests.{name}"
        if modname not in sys.modules:
            result = f"'{name}' is not a valid test.  Try"
            for mname in sys.modules:
                if mname.startswith("flacarray.tests."):
                    short_name = mname.replace("flacarray.tests.", "")
                    result += f"\n  - '{short_name}'"
            result += "\n"
            raise RuntimeError(result)
        suite.addTest(loader.loadTestsFromModule(sys.modules[modname]))
    else:
        # Load all the tests
        suite.addTest(loader.loadTestsFromModule(test_bindings))
        suite.addTest(loader.loadTestsFromModule(test_utils))
        suite.addTest(loader.loadTestsFromModule(test_array))
        suite.addTest(loader.loadTestsFromModule(test_hdf5))
        suite.addTest(loader.loadTestsFromModule(test_zarr))

    ret = 0
    _ret = runner.run(suite)
    if not _ret.wasSuccessful():
        ret = 1
        if use_mpi:
            MPI.COMM_WORLD.Abort(6)

    if ret > 0:
        print("Some tests failed", flush=True)
        sys.exit(6)

    return ret
