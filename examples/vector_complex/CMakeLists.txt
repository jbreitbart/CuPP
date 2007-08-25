# Add current directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )

# Link cuda code in a library to something else.
CUDA_ADD_LIBRARY(kernel_vector_complex kernel_vector.cu )

#list all source files here
ADD_EXECUTABLE(vector_complex_example vector.cpp)

#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(vector_complex_example cupp kernel_vector_complex cuda cudart GL GLU)

