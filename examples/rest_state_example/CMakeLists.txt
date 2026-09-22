set(REST_STATE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/rest_state.h)

add_executable(generate_rest_state generate_rest_state.cpp)
target_include_directories(generate_rest_state PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(generate_rest_state PUBLIC RTNeural)
target_compile_features(generate_rest_state PRIVATE cxx_std_17)
target_compile_definitions(generate_rest_state PRIVATE RTNEURAL_ROOT_DIR="${PROJECT_SOURCE_DIR}/")

add_custom_command(OUTPUT ${REST_STATE_HEADER}
                   COMMAND generate_rest_state ${REST_STATE_HEADER}
                   DEPENDS generate_rest_state ${PROJECT_SOURCE_DIR}/models/full_model.json
                   COMMENT "Generating rest state header")

create_example(rest_state_example)
target_sources(rest_state_example PRIVATE ${REST_STATE_HEADER})
target_include_directories(rest_state_example PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(rest_state_example PRIVATE RTNEURAL_ROOT_DIR="${PROJECT_SOURCE_DIR}/")
