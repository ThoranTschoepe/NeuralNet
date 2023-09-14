cmake_minimum_required(VERSION 3.24)

project(noise_function)

set(CMAKE_CXX_STANDARD 20)

add_executable(noise_function noise_function.cpp)

target_link_libraries(noise_function NeuralNet)

target_compile_definitions(noise_function PRIVATE PROGRAM_DIR="${CMAKE_CURRENT_LIST_DIR}")