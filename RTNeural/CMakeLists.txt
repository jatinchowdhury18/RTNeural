add_library(RTNeural STATIC
    gru.h
    gru.cpp
    gru_eigen.h
    gru_eigen.cpp
    gru_xsimd.h
    gru_xsimd.cpp
    dense.h
    dense_eigen.h
    dense_xsimd.h
    activation.h
    Model.h
    Layer.h
)

# -fpic
set_property(TARGET RTNeural PROPERTY POSITION_INDEPENDENT_CODE ON)

# external libraries
target_include_directories(RTNeural PUBLIC ../modules/Eigen)
target_include_directories(RTNeural PUBLIC ../modules/xsimd/include)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
