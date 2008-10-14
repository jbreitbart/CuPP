SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/)

# add device to library
ADD_LIBRARY(cupp SHARED device.cpp)

# generate make install
INSTALL(TARGETS cupp LIBRARY DESTINATION lib)

# set minimum cmake version
cmake_minimum_required(VERSION 2.4)
