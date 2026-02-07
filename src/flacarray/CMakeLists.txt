
# Create an object library of the compiled sources.

set(libflcarr_SOURCES
    libflacarray/compress.c
    libflacarray/decompress.c
    libflacarray/utils.c
)

set(libflcarr_HEADERS
    libflacarray/flacarray.h
)

add_library(flcarr OBJECT ${libflcarr_SOURCES})

target_compile_options(flcarr PRIVATE "${OpenMP_C_FLAGS}")

target_include_directories(flcarr BEFORE PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/libflacarray"
)

target_include_directories(flcarr PUBLIC "${FLAC_INCLUDE_DIRS}")
target_link_libraries(flcarr PUBLIC "${FLAC_LIBRARIES}")

target_link_libraries(flcarr PUBLIC "${OpenMP_C_LIBRARIES}")

# Create an actual installed library, usable by external compiled software

add_library(flacarray SHARED)
target_link_libraries(flacarray PRIVATE flcarr)
set_target_properties(flacarray PROPERTIES PUBLIC_HEADER ${libflcarr_HEADERS})

install(TARGETS flacarray
    LIBRARY DESTINATION "${SKBUILD_DATA_DIR}/lib"
    PUBLIC_HEADER DESTINATION "${SKBUILD_DATA_DIR}/include"
)

# Create a compiled python extension

cython_transpile(_libflacarray.pyx LANGUAGE C OUTPUT_VARIABLE libflacarray_c)

Python_add_library(_libflacarray MODULE "${libflacarray_c}" WITH_SOABI)

target_link_libraries(_libflacarray PRIVATE flcarr)

target_include_directories(_libflacarray PRIVATE ${NUMPY_INCLUDE_DIR})

install(TARGETS _libflacarray DESTINATION ${SKBUILD_PROJECT_NAME})

# Install python files

set(PYTHON_SOURCES
    __init__.py
    array.py
    compress.py
    decompress.py
    demo.py
    hdf5_load_v0.py
    hdf5_load_v1.py
    hdf5_utils.py
    hdf5.py
    io_common.py
    mpi.py
    utils.py
    zarr_load_v0.py
    zarr_load_v1.py
    zarr.py
)

install(FILES ${PYTHON_SOURCES} DESTINATION ${SKBUILD_PROJECT_NAME})

add_subdirectory(scripts)
add_subdirectory(tests)
