include_directories(../RTNeural)

add_executable(rtneural_tests tests.cpp)
target_link_libraries(rtneural_tests LINK_PUBLIC RTNeural)

add_custom_command(TARGET rtneural_tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:rtneural_tests> to ${PROJECT_BINARY_DIR}/rtneural_tests"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rtneural_tests> ${PROJECT_BINARY_DIR}/rtneural_tests)

#copy models for testing to build directory post build command
add_custom_command(TARGET rtneural_tests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "copying models to ${PROJECT_BINARY_DIR}/models"
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/../models ${PROJECT_BINARY_DIR}/models)

#copy test_data folder to build directory post build command
add_custom_command(TARGET rtneural_tests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "copying test_data to ${PROJECT_BINARY_DIR}/test_data"
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/../test_data ${PROJECT_BINARY_DIR}/test_data)

#copy models for testing to build directory for tests post build command
add_custom_command(TARGET rtneural_tests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "copying models to ${PROJECT_BINARY_DIR}/models"
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/../models ${PROJECT_BINARY_DIR}/tests/models)

#copy test_data folder to build directory for tests post build command
add_custom_command(TARGET rtneural_tests
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "copying test_data to ${PROJECT_BINARY_DIR}/test_data"
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/../test_data ${PROJECT_BINARY_DIR}/tests/test_data)

option(RTNEURAL_CODE_COVERAGE "Build RTNeural tests with code coverage flags" OFF)
if(RTNEURAL_CODE_COVERAGE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/EnableCoverageFlags.cmake)
    enable_coverage_flags(rtneural_tests)
endif()

