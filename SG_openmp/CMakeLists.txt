cmake_minimum_required(VERSION 3.24)
project(SG)

set(GCC_COVERAGE_COMPILE_FLAGS "-O2 -flto -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

find_package(OpenMP)

add_executable(SG SG.cpp)
