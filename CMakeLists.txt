# Created by SwiftWare Lab on 9/24.
# CE 4SP4 - High Performance Programming
# Copyright (c) 2024 SwiftWare Lab

cmake_minimum_required(VERSION 3.18)

option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)

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
        GIT_TAG v1.9.0
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
if (PROFILING_ENABLED)
    # enable PFM
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()



FetchContent_MakeAvailable(googlebenchmark)


project(multithreading C CXX )

# find openmp
find_package(OpenMP REQUIRED)


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
