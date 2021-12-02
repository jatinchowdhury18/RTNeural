# enable_coverage_flags(<target-name>)
#
# Enables code coverage flags for this target
function(enable_coverage_flags target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Add required flags (GCC & LLVM/Clang)
        message(STATUS "${target} -- Appending code coverage compiler flags: -O0 -g --coverage")
        target_compile_options(${target} PUBLIC
                -O0        # no optimization
                -g         # generate debug info
                --coverage # sets all required flags
                )

        target_link_options(${target} PUBLIC --coverage)
    endif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
endfunction(enable_coverage_flags)
