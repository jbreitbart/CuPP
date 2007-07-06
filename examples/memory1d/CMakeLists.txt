# Add current directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(kernel kernel.cu )

#list all source files here
ADD_EXECUTABLE(memory1d main.cpp)

#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(memory1d cupp kernel cuda cudart GL GLU)

