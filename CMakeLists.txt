cmake_minimum_required(VERSION 3.21)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

# Auxiliary files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
    message(WARNING "\
    This CMake file is meant to be executed using 'scikit-build-core'.
    Running it directly will almost certainly not produce the desired
    result. If you are a user trying to install this package, use the
    command below, which will install all necessary build dependencies,
    compile the package in an isolated environment, and then install it.

    $ pip install .

    ")
endif()

# We are building libraries that will eventually be linked into shared
# modules.  All code should be built with PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use modern C
set(CMAKE_C_STANDARD 11)

# For installing the stand-alone library and header
include(GNUInstallDirs)

# OpenMP
if(NOT DISABLE_OPENMP)
    find_package(OpenMP)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
include(UseCython)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include(), end='')"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
)

find_package(FLAC REQUIRED)

if(DEFINED FLAC_VERSION)
    if(FLAC_VERSION STREQUAL "")
        message(STATUS "Cannot determine FLAC version- assuming it is >= 1.4.0")
    else()
        string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\..*" "\\1"
            FLAC_MAJ_VERSION "${FLAC_VERSION}")
        string(REGEX REPLACE "^[0-9]+\\.([0-9]+)\\..*" "\\1"
            FLAC_MIN_VERSION "${FLAC_VERSION}")
        if(FLAC_MAJ_VERSION GREATER 1)
            # Future proofing
            message(STATUS "Found FLAC version ${FLAC_VERSION}")
        else()
            if(FLAC_MIN_VERSION GREATER_EQUAL 4)
                message(STATUS "Found FLAC version ${FLAC_VERSION}")
            else()
                message(FATAL_ERROR "FLAC version ${FLAC_VERSION} is not >= 1.4.0")
            endif()
        endif()
    endif()
else()
    message(STATUS "Cannot determine FLAC version- assuming it is >= 1.4.0")
endif()

add_subdirectory(src)
