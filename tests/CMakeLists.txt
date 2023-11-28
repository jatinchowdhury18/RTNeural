add_subdirectory(unit)
add_subdirectory(functional)

option(RTNEURAL_CODE_COVERAGE "Build RTNeural tests with code coverage flags" OFF)
if(RTNEURAL_CODE_COVERAGE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/EnableCoverageFlags.cmake)
    enable_coverage_flags(rtneural_test_unit)
    enable_coverage_flags(rtneural_test_functional)
endif()
