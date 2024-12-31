#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will run in a manylinux2014 (CentOS 7) container.
#

set -e

prefix=$1

if [ "x${prefix}" = "x" ]; then
    prefix=/usr/local
fi

export PREFIX="${prefix}"

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
echo "Wheel script directory = ${scriptdir}"

# Build options
export CC=gcc
export CFLAGS="-O3 -fPIC -pthread"
export OMPFLAGS="-fopenmp"

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v cmake wheel

# In order to maximize ABI compatibility with numpy, build with the newest numpy
# version containing the oldest ABI version compatible with the python we are using.
# NOTE: for now, we build with numpy 2.0.x, which is backwards compatible with
# numpy-1.x and forward compatible with numpy-2.x.
pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
# if [ ${pyver} == "3.8" ]; then
#     numpy_ver="1.20"
# fi
# if [ ${pyver} == "3.9" ]; then
#     numpy_ver="1.20"
# fi
# if [ ${pyver} == "3.10" ]; then
#     numpy_ver="1.22"
# fi
# if [ ${pyver} == "3.11" ]; then
#     numpy_ver="1.24"
# fi
numpy_ver="2.0.1"

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v -r "${scriptdir}/build_requirements.txt" "numpy==${numpy_ver}"

# Build compiled dependencies

# For testing locally
# export MAKEJ=8
export MAKEJ=2
export STATIC="${static}"
export SHLIBEXT="so"
export CLEANUP=yes

flac_version=1.4.3
flac_dir=flac-${flac_version}
flac_pkg=${flac_dir}.tar.gz

echo "Fetching libFLAC..."

if [ ! -e ${flac_pkg} ]; then
    curl -SL "https://github.com/xiph/flac/archive/refs/tags/${flac_version}.tar.gz" -o "${flac_pkg}"
fi

echo "Building libFLAC..."

rm -rf ${flac_dir}
tar xzf ${flac_pkg} \
    && pushd ${flac_dir} >/dev/null 2>&1 \
    && mkdir -p build \
    && pushd build >/dev/null 2>&1 \
    && cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DBUILD_DOCS=OFF \
    -DWITH_OGG=OFF \
    -DBUILD_CXXLIBS=OFF \
    -DBUILD_PROGRAMS=OFF \
    -DBUILD_UTILS=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DINSTALL_MANPAGES=OFF \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    .. \
    && make -j ${MAKEJ} install \
    && popd >/dev/null 2>&1 \
    && popd >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${flac_dir}
fi
