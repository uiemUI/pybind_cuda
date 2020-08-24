cmake_minimum_required(VERSION 3.8)
project(TestD )
enable_language(CUDA)

find_package(CUDA REQUIRED)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   -Xcompiler -fPIC -G -g ")
add_executable(TestD kernel.h kernel.cu)
target_link_libraries(TestD cublas)
