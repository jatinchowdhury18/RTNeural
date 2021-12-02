cmake_minimum_required(VERSION 3.8)
project(RTNeural VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(RTNeural)

include(cmake/SIMDExtensions.cmake)
include(cmake/ChooseBackend.cmake)

option(BUILD_TESTS "Build RTNeural accuracy tests" OFF)
if(BUILD_TESTS)
    message(STATUS "RTNeural -- Configuring tests...")
    add_subdirectory(tests)
endif()

option(BUILD_BENCH "Build RTNeural benchmarks" OFF)
if(BUILD_BENCH)
    message(STATUS "RTNeural -- Configuring benchmarks...")
    add_subdirectory(bench)
endif()

option(BUILD_EXAMPLES "Build RTNeural examples" OFF)
if(BUILD_EXAMPLES)
    message(STATUS "RTNeural -- Configuring examples...")
    add_subdirectory(examples)
endif()
