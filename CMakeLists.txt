cmake_minimum_required(VERSION 3.24)
project(SG)


set(GCC_COVERAGE_COMPILE_FLAGS "-flto -O2 -march=native -lcblas")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

find_package(BLAS REQUIRED)

add_executable(SG SG.cpp)
