
# Add current directory and the cupp include directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} ../../include )

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(kernel kernel.cu )

#add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -O2 -g)

#CuPP
INCLUDE_DIRECTORIES(../../include)
LINK_DIRECTORIES(../../lib)

#CUDA
INCLUDE_DIRECTORIES(/usr/local/cuda/include/)
LINK_DIRECTORIES(/usr/local/cuda/lib)

#list all source files here
ADD_EXECUTABLE(memory1d main.cpp)

#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(memory1d cupp kernel cuda cudart GL GLU)

