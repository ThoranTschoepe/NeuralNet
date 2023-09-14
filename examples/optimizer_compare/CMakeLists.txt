cmake_minimum_required(VERSION 3.24)

project(optimizer_compare)

set(CMAKE_CXX_STANDARD 20)

add_executable(optimizer_compare optimizer_compare.cpp)

target_link_libraries(optimizer_compare NeuralNet)

target_compile_definitions(optimizer_compare PRIVATE PROGRAM_DIR="${CMAKE_CURRENT_LIST_DIR}")
