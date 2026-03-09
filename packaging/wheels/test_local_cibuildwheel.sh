#!/bin/bash

# Before running this from the git checkout directory,
# you should pip install cibuildwheel.
#
# If the build dies, you can poke around in the residual container:
#
#   docker ps -a (to find the ID)
#   docker start <ID>
#   docker exec -it <ID> /bin/bash
#

export CIBW_BUILD="cp313-manylinux_x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="manylinux_2_28"
export CIBW_BUILD_VERBOSITY=3

# Uncomment to leave the container for debugging
#export CIBW_DEBUG_KEEP_CONTAINER=1

export CIBW_ENVIRONMENT_LINUX=""
export CIBW_BEFORE_BUILD_LINUX=./packaging/wheels/install_deps_linux.sh
export CIBW_BEFORE_TEST="export OMP_NUM_THREADS=2"
export CIBW_TEST_REQUIRES="h5py zarr"
export CIBW_TEST_COMMAND="python -c 'import flacarray.tests; flacarray.tests.run()'"

# Get the current date for logging
now=$(date "+%Y-%m-%d_%H:%M:%S")

# Run it
cibuildwheel --platform linux --output-dir wheelhouse . 2>&1 | tee log_${now}

