SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/)

# if you don't want the full compiler output, remove the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)

#add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -O2)


ADD_LIBRARY(cupp SHARED device.cpp)

