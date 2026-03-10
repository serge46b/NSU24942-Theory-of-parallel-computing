cmake_minimum_required(VERSION 3.10)
project(openMP1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2") # Значения по умолчанию

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(main main.cpp)
