include_directories(../src)
include_directories(../modules/Eigen)
include_directories(../modules/json)

# DENSE test
add_executable(dense dense.cpp)
target_link_libraries(dense LINK_PUBLIC MLUtils)

add_custom_command(TARGET dense
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:dense> to ${PROJECT_BINARY_DIR}/dense"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:dense> ${PROJECT_BINARY_DIR}/dense)

    # GRU test
add_executable(gru gru.cpp)
target_link_libraries(gru LINK_PUBLIC MLUtils)

add_custom_command(TARGET gru
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:gru> to ${PROJECT_BINARY_DIR}/gru"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gru> ${PROJECT_BINARY_DIR}/gru)
