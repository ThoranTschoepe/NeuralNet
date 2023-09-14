cmake_minimum_required(VERSION 3.24)

project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(NeuralNet ${SOURCES})

target_include_directories(NeuralNet PUBLIC include)


if (PROJECT_IS_TOP_LEVEL)
    add_subdirectory(examples)
endif ()