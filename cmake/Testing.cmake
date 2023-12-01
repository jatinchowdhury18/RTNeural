option(RTNEURAL_TEST_REPORTS "Output test reports to XML files" OFF)

macro(rtneural_setup_testing)

    include(CTest)
    enable_testing()
    add_custom_target(rtneural_test COMMAND ctest -C $<CONFIG> --output-on-failure)

    # From the GoogleTest README:
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    CPMAddPackage("gh:google/googletest@1.14.0")

endmacro()

function(rtneural_add_test)

    set(one_val_args TARGET)
    set(multi_val_args SOURCES DEPENDENCIES)
    cmake_parse_arguments(arg "" "${one_val_args}" "${multi_val_args}" ${ARGN})

    add_executable(${arg_TARGET} ${arg_SOURCES})
    target_link_libraries(${arg_TARGET} PUBLIC gtest_main gmock ${arg_DEPENDENCIES})
    target_compile_definitions(${arg_TARGET} PRIVATE RTNEURAL_ROOT_DIR="${CMAKE_SOURCE_DIR}/")

    if(RTNEURAL_TEST_REPORTS)
        set(test_cmd_args --gtest_output=xml:${arg_TARGET}_report.xml)
    endif()

    add_test(
        NAME ${arg_TARGET}
        WORKING_DIRECTORY ./
        COMMAND ${arg_TARGET} ${test_cmd_args})

    add_dependencies(rtneural_test ${arg_TARGET})

endfunction()
