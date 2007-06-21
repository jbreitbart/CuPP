
PROJECT(cupp)

#if you don't want the full compiler output, remove the following line

SET(CMAKE_VERBOSE_MAKEFILE ON)

# INCLUDE AND LINK WITH CUDA
INCLUDE_DIRECTORIES(/usr/local/cuda/include/)
LINK_DIRECTORIES(/usr/local/cuda/lib)


SUBDIRS(src) #build the libraries

SUBDIRS(examples/memory1d)
