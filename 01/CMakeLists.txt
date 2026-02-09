cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(test_proj VERSION 0.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DOUBLE "Set type of output array to double instead of float" OFF)
if(USE_DOUBLE)
  add_definitions(-DUSE_DOUBLE)
endif()

add_executable(main main.cpp)
