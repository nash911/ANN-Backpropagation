#-------CMakeList for ANN-Backpropagation Project-----------#

cmake_minimum_required (VERSION 2.6.0)
PROJECT(ANN_Backpropagation)

add_executable(ANN_Backpropagation
  Source/main.cpp
  Source/data.cpp
  Source/neural_network.cpp
)

### executable
target_link_libraries(ANN_Backpropagation -g -O2 -larmadillo)

