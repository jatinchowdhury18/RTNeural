option(RTNEURAL_EIGEN "Use Eigen library for vector operations" OFF)
option(RTNEURAL_XSIMD "Use xsimd library for vector operations" OFF)
option(RTNEURAL_STL "Use STL for all operations" OFF)
if(RTNEURAL_EIGEN)
    message(STATUS "RTNeural -- Using Eigen backend")
    target_compile_definitions(RTNeural PUBLIC RTNEURAL_USE_EIGEN=1)
    target_include_directories(RTNeural PUBLIC modules/Eigen)
elseif(RTNEURAL_XSIMD)
    message(STATUS "RTNeural -- Using xsimd backend")
    target_compile_definitions(RTNeural PUBLIC RTNEURAL_USE_XSIMD=1)
    target_include_directories(RTNeural PUBLIC modules/xsimd/include)
elseif(RTNEURAL_STL)
    message(STATUS "RTNeural -- Using STL backend")
else()
    message(STATUS "RTNeural -- Using Eigen backend (Default)")
    target_compile_definitions(RTNeural PUBLIC RTNEURAL_USE_EIGEN=1)
    target_include_directories(RTNeural PUBLIC modules/Eigen)
endif()
