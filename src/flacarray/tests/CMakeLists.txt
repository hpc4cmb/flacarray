# Install python files

set(PYTHON_SOURCES
    __init__.py
    array.py
    bindings.py
    hdf5.py
    runner.py
    utils.py
    zarr.py
)

install(FILES ${PYTHON_SOURCES} DESTINATION "${SKBUILD_PROJECT_NAME}/tests/")
