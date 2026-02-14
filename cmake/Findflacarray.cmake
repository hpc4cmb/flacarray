# - Find the flacarray C library
#
# First version:
#   Copyright (c) 2026, Ted Kisner
#
# Usage:
#   find_package(flacarray [REQUIRED])
#
# It sets the following variables:
#   flacarray_FOUND                  ... true if flacarray is found on the system
#   flacarray_LIBRARIES              ... full path to flacarray libraries
#   flacarray_INCLUDE_DIRS           ... flacarray include directory paths
#
# This file uses the flacarray_config script, which should be in the executable
# search path if flacarray has been installed properly in the current environment.
#
# The following variables will be checked by the function
#   flacarray_ROOT                   ... ignore flacarray_config script and look here.
#   flacarray_USE_STATIC_LIBS        ... search for static versions of the libs
#

# Check whether to search static or dynamic libs

if(${flacarray_USE_STATIC_LIBS})
    set(FLACARRAY_LIB_NAME "flacarray_static")
else()
    set(FLACARRAY_LIB_NAME "flacarray")
endif()

if(flacarray_ROOT)
    # Find libs
    find_library(
        FLACARRAY_LIB
        NAMES ${FLACARRAY_LIB_NAME}
        PATHS ${flacarray_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
    )
    # Find includes
    find_path(flacarray_INCLUDE_DIRS
        NAMES "flacarray.h"
        PATHS ${flacarray_ROOT}
        PATH_SUFFIXES "include"
        NO_DEFAULT_PATH
    )
else()
    # Use flacarray_config
    find_program(FLACARRAY_CONFIG flacarray_config REQUIRED)
    execute_process(
        COMMAND "${FLACARRAY_CONFIG}" --package
        OUTPUT_VARIABLE flacarray_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${FLACARRAY_CONFIG}" --include
        OUTPUT_VARIABLE flacarray_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${FLACARRAY_CONFIG}" --libraries
        OUTPUT_VARIABLE FLACARRAY_LIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(FLACARRAY_LIB)
    set(flacarray_LIBRARIES ${FLACARRAY_LIB})
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(flacarray
    REQUIRED_VARS flacarray_INCLUDE_DIRS FLACARRAY_LIB
    HANDLE_COMPONENTS
)

mark_as_advanced(
    flacarray_INCLUDE_DIRS
    flacarray_LIBRARIES
)
