cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(task2_server LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")
find_package(Threads REQUIRED)

add_executable(task2_main main.cpp)
target_link_libraries(task2_main PRIVATE Threads::Threads)

add_executable(task2_test test_verify.cpp)
