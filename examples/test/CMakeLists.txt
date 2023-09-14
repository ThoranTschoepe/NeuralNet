cmake_minimum_required(VERSION 3.24)

project(test)

set(CMAKE_CXX_STANDARD 20)

add_executable(test test.cpp)

target_link_libraries(test NeuralNet)

target_compile_definitions(test PRIVATE PROGRAM_DIR="${CMAKE_CURRENT_LIST_DIR}")
