# Compiled Library

This directory contains sources for the `libflacarray` compiled library.
There are some standard C source files and the output library is wrapped by
Cython in the parent directory.

You should not have to modify any files in this directory unless you are working
with the code that directly calls libFLAC.

## Developer Notes

If you are hacking on the C source files, it can be convenient to compile and
test those manually independent of the overall package build system. There is a
small stand-alone test script in this directory which you can use. Edit the
`low_level.mk` make file and change any compiler options (for example, disable
optimization and enable debug symbols). Then build the test script:

    make -f low_level.mk

and then run the `test_low_level` executable. Since this is using low-level C
memory management and pointer math, it is always good to run through valgrind
and verify correct memory handling. For example, if you are building in a conda
development environment, you may have to add the conda prefix to
`LD_LIBRARY_PATH` in order to load libFLAC at runtime:

    OMP_NUM_THREADS=4 \
    LD_LIBRARY_PATH=${CONDA_PREFIX}/lib \
    valgrind \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    ./test_low_level 2>&1 | tee log
