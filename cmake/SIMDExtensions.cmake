option(RTNEURAL_USE_AVX2 "Enables AVX2 SIMD Support" OFF)
if(NOT RTNEURAL_USE_AVX2)
    target_compile_definitions(RTNeural PUBLIC RTNEURAL_DEFAULT_ALIGNMENT=16)
else()
    message(STATUS "Attempting to enable AVX2...")

    include(CheckCXXCompilerFlag)
    if(MSVC)
        CHECK_CXX_COMPILER_FLAG("/arch:AVX2" COMPILER_OPT_ARCH_AVX_SUPPORTED)
        if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
            message(STATUS "AVX2 flags enabled for MSVC!")
            target_compile_options(RTNeural PUBLIC /arch:AVX2)
            target_compile_definitions(RTNeural PUBLIC RTNEURAL_AVX2_ENABLED=1)
            target_compile_definitions(RTNeural PUBLIC RTNEURAL_DEFAULT_ALIGNMENT=32)
        else()
            message(STATUS "Unable to add AVX2 flags for MSVC!")
            target_compile_definitions(RTNeural PUBLIC RTNEURAL_DEFAULT_ALIGNMENT=16)
        endif()
    else()
        CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
        if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
            message(STATUS "AVX2 flags enabled for ${CMAKE_CXX_COMPILER_ID}!")
            target_compile_options(RTNeural PUBLIC -march=native)
            target_compile_definitions(RTNeural PUBLIC RTNEURAL_AVX2_ENABLED=1)
            target_compile_definitions(RTNeural PUBLIC RTNEURAL_DEFAULT_ALIGNMENT=32)
        else()
            message(STATUS "Unable to enable AVX2 flags for ${CMAKE_CXX_COMPILER_ID}!")
            target_compile_definitions(RTNeural PUBLIC RTNEURAL_DEFAULT_ALIGNMENT=16)
        endif()
    endif()
endif()
