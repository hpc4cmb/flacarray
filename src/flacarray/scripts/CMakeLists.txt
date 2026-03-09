# Install python files

set(CONF_EXTRA_LIBS "${FLAC_LIBRARIES}")
if(OpenMP_C_FOUND)
    set(CONF_EXTRA_FLAGS "${OpenMP_C_FLAGS}")
else()
    set(CONF_EXTRA_FLAGS "")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.py.in" # Input file path
    "${CMAKE_CURRENT_BINARY_DIR}/config.py"    # Output file path
    @ONLY                                      # Only @VAR@ form only
)

set(PYTHON_SOURCES
    __init__.py
    benchmark.py
    "${CMAKE_CURRENT_BINARY_DIR}/config.py"
)

install(FILES ${PYTHON_SOURCES} DESTINATION "${SKBUILD_PROJECT_NAME}/scripts")
