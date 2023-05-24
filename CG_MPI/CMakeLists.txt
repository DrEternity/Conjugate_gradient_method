cmake_minimum_required(VERSION 3.10)
project(CG)

# Find MPI package
find_package(MPI REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -march=native")

# Create executable
add_executable(CG CG.cpp)

# Link MPI libraries
target_link_libraries(CG MPI::MPI_CXX)

