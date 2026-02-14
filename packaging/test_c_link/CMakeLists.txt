cmake_minimum_required(VERSION 3.21)

# Auxiliary files, including Findflacarray.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Add CTest
enable_testing()

# Define the project with name and version from pyproject.toml
project(c_test LANGUAGES C VERSION 1.0)

# Use modern C
set(CMAKE_C_STANDARD 11)

# Find libflacarray
find_package(flacarray REQUIRED)

# Create the compiled test executable

add_executable(test_c_link test.c)
target_link_libraries(test_c_link PRIVATE ${flacarray_LIBRARIES})
target_include_directories(test_c_link BEFORE PUBLIC ${flacarray_INCLUDE_DIRS})

# Add a test that runs this executable
add_test(
    NAME test_link
    COMMAND test_c_link
)
