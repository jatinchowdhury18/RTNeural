add_executable(hello_rtneural hello_rtneural.cpp)

target_include_directories(hello_rtneural PUBLIC ${CMAKE_SOURCE_DIR})

add_custom_command(TARGET hello_rtneural
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:hello_rtneural> to ${EXAMPLES_OUT_DIR}/hello_rtneural"
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hello_rtneural> ${EXAMPLES_OUT_DIR}/hello_rtneural)
