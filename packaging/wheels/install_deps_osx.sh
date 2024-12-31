#!/bin/bash
#
# This script is designed to run within a container managed by cibuildwheel.
# This will use a recent version of OS X.
#

set -e

prefix=$1

if [ "x${prefix}" = "x" ]; then
    prefix=/usr/local
fi

export PREFIX="${prefix}"

# Make sure lib64 points to lib, so that our dependencies can be found
sudo rm -rf "${prefix}/lib64"
sudo ln -s "${PREFIX}/lib" "${PREFIX}/lib64"

# If we are running on github CI, ensure that permissions
# are set on /usr/local.  See:
# https://github.com/actions/runner-images/issues/9272
sudo chown -R runner:admin /usr/local/

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1
echo "Wheel script directory = ${scriptdir}"

# Are we using gcc?  Useful for OpenMP.
# use_gcc=yes
use_gcc=no
gcc_version=14

# Build options.

if [ "x${use_gcc}" = "xyes" ]; then
    export CC=gcc-${gcc_version}
    export CFLAGS="-O3 -fPIC"
    export OMPFLAGS="-fopenmp"
else
    # Set the deployment target based on how python was built
    export MACOSX_DEPLOYMENT_TARGET=$(python3 -c "import sysconfig as s; print(s.get_config_vars()['MACOSX_DEPLOYMENT_TARGET'])")
    echo "Using MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
    export CC=clang
    export CFLAGS="-O3 -fPIC"
    export OMPFLAGS=""
fi

# Force uninstall flac tools, to avoid conflicts with our
# custom compiled version.
brew uninstall -f --ignore-dependencies flac libogg libsndfile libvorbis opusfile sox
if [ "x${use_gcc}" = "xyes" ]; then
    brew install gcc@${gcc_version}
fi

# Update pip
pip install --upgrade pip

# Install a couple of base packages that are always required
pip install -v wheel

# In order to maximize ABI compatibility with numpy, build with the newest numpy
# version containing the oldest ABI version compatible with the python we are using.
# NOTE: for now, we build with numpy 2.0.x, which is backwards compatible with
# numpy-1.x and forward compatible with numpy-2.x.
pyver=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
# if [ ${pyver} == "3.8" ]; then
#     numpy_ver="1.20"
# fi
# if [ ${pyver} == "3.9" ]; then
#     numpy_ver="1.24"
# fi
# if [ ${pyver} == "3.10" ]; then
#     numpy_ver="1.24"
# fi
# if [ ${pyver} == "3.11" ]; then
#     numpy_ver="1.24"
# fi
numpy_ver="2.0.1"

# Install build requirements.
CC="${CC}" CFLAGS="${CFLAGS}" pip install -v -r "${scriptdir}/../pip_build_requirements.txt" "numpy==${numpy_ver}"

# Build compiled dependencies

export MAKEJ=2
export STATIC=no
export SHLIBEXT="dylib"
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
    -DENABLE_MULTITHREADING=ON \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    .. \
    && make -j ${MAKEJ} install \
    && popd >/dev/null 2>&1 \
    && popd >/dev/null 2>&1

if [ "x${CLEANUP}" = "xyes" ]; then
    rm -rf ${flac_dir}
fi
