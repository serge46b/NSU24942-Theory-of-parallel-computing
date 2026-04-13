cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(matrix_vector_threads LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(Threads REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Threads::Threads)
