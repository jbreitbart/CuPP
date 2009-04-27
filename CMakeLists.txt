PROJECT(cupp)

# load cuda macros
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCUDA.cmake)

# include and link with CUDA
INCLUDE_DIRECTORIES(${CUDA_INSTALL_PREFIX}/include/)
LINK_DIRECTORIES(${CUDA_INSTALL_PREFIX}/lib)

# set the include path
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)
CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)

# if you don want the full compiler output, uncomment the following line
#SET(CMAKE_VERBOSE_MAKEFILE ON)

# add definitions, compiler switches, etc.
#ADD_DEFINITIONS(-Wall -W -O2 -g)

# build the CuPP library
SUBDIRS(src)

# build the examples
SUBDIRS(examples)

# generate make install
INSTALL(DIRECTORY include/cupp DESTINATION include)

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
