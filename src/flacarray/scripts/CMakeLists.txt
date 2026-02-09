# Install python files

set(PYTHON_SOURCES
    __init__.py
    benchmark.py
    config.py
)

install(FILES ${PYTHON_SOURCES} DESTINATION "${SKBUILD_PROJECT_NAME}/scripts")
