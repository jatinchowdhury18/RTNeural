set(EXAMPLES_OUT_DIR ${PROJECT_BINARY_DIR}/examples_out)
file(MAKE_DIRECTORY ${EXAMPLES_OUT_DIR})

function(create_example example_name)
    message(STATUS "RTNeural -- Configuring example: ${example_name}")
    add_executable(${example_name} ${example_name}.cpp)
    target_include_directories(${example_name} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${example_name} PUBLIC RTNeural)
    target_compile_features(${example_name} PRIVATE cxx_std_17)

    add_custom_command(TARGET ${example_name}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:${example_name}> to ${EXAMPLES_OUT_DIR}/${example_name}"
                       COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${example_name}> ${EXAMPLES_OUT_DIR}/${example_name})
endfunction()

add_subdirectory(hello_rtneural)
add_subdirectory(rtneural_static_model)
add_subdirectory(rtneural_dynamic_model)
add_subdirectory(custom_layer_model)
add_subdirectory(torch/torch_conv1d)
add_subdirectory(torch/torch_gru)
