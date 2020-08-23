cmake_minimum_required(VERSION 3.8)
project(exx)
enable_language(CUDA)
# include_directories("C:/INclude/vcpkg/installed/x64-windows/include")
# include_directories("C:/INclude/vcpkg/installed/x64-windows/include/python3.7")
# add_library(mylib SHARED exx.cpp)
# set_target_properties(mylib PROPERTIES SUFFIX ".pyd")
# add_library(kernelib SHARED kernel.h kernel.cu)

find_package(pybind11 CONFIG REQUIRED)
if(PYBIND11_FOUND)
    message("found/n")
endif(PYBIND11_FOUND)
find_package(CUDA REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# message("lujin" ${PYBIND11_INCLUDE_DIR} "/n" ${PYTHON_INCLUDE_DIRS})

# SET(ARCH "-gencode arch=compute_30,code=sm_30
#           -gencode arch=compute_35,code=sm_35
#           -gencode arch=compute_50,code=[sm_50,compute_50]
#           -gencode arch=compute_52,code=[sm_52,compute_52]")
# SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}   -Xcompiler --compiler-options '-fPIC' ${ARCH}")


add_library(kernelib SHARED kernel.h kernel.cu)
pybind11_add_module(mylib SHARED kernel.h  mylib.cpp)

set_target_properties(mylib kernelib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(mylib PRIVATE kernelib)
