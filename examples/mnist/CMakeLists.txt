cmake_minimum_required(VERSION 3.24)

project(mnist)

include_directories(mnist-master/include)
add_executable(mnist main.cpp)

target_link_libraries(mnist NeuralNet)

target_compile_features(mnist PRIVATE cxx_range_for)

target_compile_definitions(mnist PRIVATE PROGRAM_DIR="${CMAKE_CURRENT_LIST_DIR}")
