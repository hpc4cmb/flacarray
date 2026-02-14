#!/bin/sh

cmake_version=$1

c_file="libflacarray/version.c"
c_version=""
if [ -f "${c_file}" ]; then
    # Existing file, read in version
    c_version=`cat "${c_file}" | sed -e 's/.*= "\(.*\)".*/\1/'`
fi

if [ "x${c_version}" != "x${cmake_version}" ]; then
    # The version has changed, update the file
    echo "const char* FLACARRAY_VERSION = \"${cmake_version}\";" > "${c_file}"
fi

py_file="_version.py"
py_version=""
if [ -f "${py_file}" ]; then
    # Existing file, read in version
    py_version=`cat "${py_file}" | sed -e 's/.*= "\(.*\)".*/\1/'`
fi

if [ "x${py_version}" != "x${cmake_version}" ]; then
    # The version has changed, update the file
    echo "__version__ = \"${cmake_version}\"" > "${py_file}"
fi
