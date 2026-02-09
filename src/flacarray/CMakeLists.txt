
# Generate the version files, if needed

set(c_version ${CMAKE_CURRENT_SOURCE_DIR}/libflacarray/version.c)
set(py_version ${CMAKE_CURRENT_SOURCE_DIR}/_version.py)

add_custom_command(OUTPUT ${c_version} ${py_version}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_version.sh ${SKBUILD_PROJECT_VERSION}
    COMMENT "Updating version files if needed ..."
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Create an object library of the compiled sources.

set(libflcarr_SOURCES
    libflacarray/compress.c
    libflacarray/decompress.c
    libflacarray/utils.c
    "${c_version}"
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
    LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
    PUBLIC_HEADER DESTINATION "${SKBUILD_PROJECT_NAME}/include"
)

# Create the compiled test executable

add_executable(flacarray_c_test libflacarray/test.c)
target_link_libraries(flacarray_c_test PRIVATE flcarr)
set_target_properties(flacarray_c_test PROPERTIES PRIVATE_HEADER ${libflcarr_HEADERS})

install(TARGETS flacarray_c_test RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR})

# Create a compiled python extension

cython_transpile(_libflacarray.pyx LANGUAGE C OUTPUT_VARIABLE libflacarray_c)

Python_add_library(_libflacarray MODULE "${libflacarray_c}" WITH_SOABI)

target_link_libraries(_libflacarray PRIVATE flcarr)

target_include_directories(_libflacarray PRIVATE ${NUMPY_INCLUDE_DIR})

install(TARGETS _libflacarray LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# Install python files

set(PYTHON_SOURCES
    __init__.py
    "${py_version}"
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
