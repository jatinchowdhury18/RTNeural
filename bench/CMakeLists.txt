include_directories(../RTNeural)

add_executable(rtneural_layer_bench layer_bench.cpp)
target_link_libraries(rtneural_layer_bench LINK_PUBLIC RTNeural)

add_custom_command(TARGET rtneural_layer_bench
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:rtneural_layer_bench> to ${PROJECT_BINARY_DIR}/rtneural_layer_bench"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rtneural_layer_bench> ${PROJECT_BINARY_DIR}/rtneural_layer_bench)

add_executable(rtneural_model_bench model_bench.cpp)
target_link_libraries(rtneural_model_bench LINK_PUBLIC RTNeural)
    
add_custom_command(TARGET rtneural_model_bench
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:rtneural_model_bench> to ${PROJECT_BINARY_DIR}/rtneural_model_bench"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rtneural_model_bench> ${PROJECT_BINARY_DIR}/rtneural_model_bench)
