cmake_minimum_required(VERSION 3.24)

project(xor)

set(CMAKE_CXX_STANDARD 20)

add_executable(xor xor.cpp)

target_link_libraries(xor NeuralNet)