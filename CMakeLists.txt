PROJECT(cupp)

# load cuda macros
FIND_PACKAGE(CUDA)
FIND_PACKAGE(boost)

#SET (CUDA_HOST_COMPILATION_CPP OFF)
#SET (CUDA_NVCC_FLAGS -D__CUDACC__)

# include and link with CUDA
IF (CMAKE_CL_64)
    SET(CUDA_LIB_PATH ${CUDA_SDK_ROOT_DIR}/lib/x64  )
ELSE (CMAKE_CL_64)
    IF (CMAKE_CL_32)
       SET(CUDA_LIB_PATH ${CUDA_SDK_ROOT_DIR}/lib )
    ELSE (CMAKE_CL_32)
       SET(CUDA_LIB_PATH ${CUDA_SDK_ROOT_DIR}/lib64)
    ENDIF(CMAKE_CL_32)
ENDIF (CMAKE_CL_64)

LINK_DIRECTORIES(${CUDA_LIB_PATH})

# set the include path

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)

# if you don want the full compiler output, uncomment the following line
#SET(CMAKE_VERBOSE_MAKEFILE ON)

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
