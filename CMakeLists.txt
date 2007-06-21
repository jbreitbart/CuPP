
PROJECT(cupp)

# SETUP THE CUDA SYSTEM
SET (CUDA_INSTALL_PREFIX /usr/local/cuda/)
SET (CUDA_BUILD_TYPE Device)

# Load cuda macros
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

# INCLUDE AND LINK WITH CUDA
INCLUDE_DIRECTORIES(/usr/local/cuda/include/)
LINK_DIRECTORIES(/usr/local/cuda/lib)


# if you don't want the full compiler output, remove the following line

SET(CMAKE_VERBOSE_MAKEFILE ON)

# build the CuPP library
SUBDIRS(src)

# build the examples
SUBDIRS(examples/memory1d)
