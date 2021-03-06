add_library(RTNeural STATIC
    activation/activation.h
    activation/activation_accelerate.h
    activation/activation_eigen.h
    activation/activation_xsimd.h
    Model.h
    Layer.h
    conv1d/conv1d.h
    conv1d/conv1d.tpp
    dense/dense.h
    dense/dense_accelerate.h
    dense/dense_eigen.h
    dense/dense_xsimd.h
    gru/gru.h
    gru/gru.tpp
    gru/gru_accelerate.h
    gru/gru_accelerate.tpp
    gru/gru_eigen.h
    gru/gru_eigen.tpp
    gru/gru_xsimd.h
    gru/gru_xsimd.tpp
    lstm/lstm.h
    lstm/lstm.tpp
    lstm/lstm_eigen.h
    lstm/lstm_eigen.tpp
    lstm/lstm_xsimd.h
    lstm/lstm_xsimd.tpp
    model_loader.h
    RTNeural.h
    RTNeural.cpp
)

# -fpic
set_property(TARGET RTNeural PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(RTNeural PROPERTIES LINKER_LANGUAGE CXX)

# external libraries
target_include_directories(RTNeural PUBLIC ../modules/Eigen)
target_include_directories(RTNeural PUBLIC ../modules/xsimd/include)
target_include_directories(RTNeural PUBLIC ../modules/json)

if(RTNEURAL_ACCELERATE)
    target_link_libraries(RTNeural PUBLIC "-framework Accelerate")
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
