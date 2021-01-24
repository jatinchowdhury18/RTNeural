cmake_minimum_required(VERSION 3.8)
project(RTNeural)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

# select library backend
option(RTNEURAL_EIGEN "Use Eigen library for vector operations" OFF)
option(RTNEURAL_STL "Use STL for all operations" OFF)
if(RTNEURAL_EIGEN)
    message("Using Eigen backend")
    add_definitions(-DUSE_EIGEN)
elseif(RTNEURAL_STL)
    message("Using STL backend")
else()
    message("Using Eigen backend (Default)")
    add_definitions(-DUSE_EIGEN)
endif()

add_subdirectory(RTNeural)

option(BUILD_TESTS "Build RTNeural accuracy tests" OFF)
if(BUILD_TESTS)
    message("Configuring tests...")
    add_subdirectory(tests)
endif()

option(BUILD_BENCH "Build RTNeural benchmarks" OFF)
if(BUILD_BENCH)
    message("Configuring benchmarks...")
    add_subdirectory(bench)
endif()
