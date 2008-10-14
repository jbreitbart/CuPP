# Add current directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(kernel_memory1d kernel_memory1d.cu )

#list all source files here
ADD_EXECUTABLE(memory1d_example memory1d.cpp)

#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(memory1d_example cupp kernel_memory1d cudart GL GLU cuda)

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
