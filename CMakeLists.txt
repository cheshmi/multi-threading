# Created by SwiftWare Lab on 9/24.
# CE 4SP4 - High Performance Programming
# Copyright (c) 2024 SwiftWare Lab

cmake_minimum_required(VERSION 3.18)

option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
option(OPENMP "Enables OpenMP" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")
include(FetchContent)

FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)


include(FetchContent)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
if (PROFILING_ENABLED)
    # enable PFM
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()



FetchContent_MakeAvailable(googlebenchmark)




# enable google test
set(BENCHMARK_ENABLE_TESTS ON)
set(BENCHMARK_ENABLE_GTEST_TESTS ON)


project(multithreading C CXX )

# find openmp
find_package(OpenMP REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set (ENV{OMP_DYNAMIC} false)

include_directories(
        ${CMAKE_SOURCE_DIR}/include)


set(CMAKE_CXX_STANDARD 17)

add_executable(vec_mul_vec
        ${CMAKE_SOURCE_DIR}/vec_mul_driver.cpp
        ${CMAKE_SOURCE_DIR}/src/vec_mul.cpp
)
target_link_libraries(vec_mul_vec
        benchmark::benchmark 
        OpenMP::OpenMP_CXX
)
