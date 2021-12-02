include_directories(../RTNeural)

add_executable(rtneural_tests tests.cpp)
target_link_libraries(rtneural_tests LINK_PUBLIC RTNeural)

add_custom_command(TARGET rtneural_tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:rtneural_tests> to ${PROJECT_BINARY_DIR}/rtneural_tests"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rtneural_tests> ${PROJECT_BINARY_DIR}/rtneural_tests)

option(RTNEURAL_CODE_COVERAGE "Build RTNeural tests with code coverage flags" OFF)
if(RTNEURAL_CODE_COVERAGE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/EnableCoverageFlags.cmake)
    enable_coverage_flags(rtneural_tests)
endif()

