cmake_minimum_required(VERSION 3.15)

# set the project name
project(cbns3d_cuda LANGUAGES CXX CUDA)

# Enable CUDA
enable_language(CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Check if CMAKE_CUDA_ARCHITECTURES is set
if (NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not defined. Please set it using -D CMAKE_CUDA_ARCHITECTURES=<value> when configuring the project.")
endif()

# Show the selected CUDA architecture
message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDA REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories("${CUDA_INCLUDE_DIRS}")

# Bring the headers into the project
include_directories(include)

# Find all *.cu files in the src/ directory
file(GLOB CU_FILES "src/*.cu")

add_library(block3d_cuda STATIC ${CU_FILES})
set_target_properties(block3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

option(INVIS_MODE "Inviscid simulation" OFF)
if (INVIS_MODE)
  target_compile_definitions(block3d_cuda
                             PRIVATE IS_INVISCID
			     )
endif()

set(SOURCES src/cbns3d.cpp src/Block3d.cpp)

# Add the executable
add_executable(cbns3d_cuda ${SOURCES})
target_link_libraries(cbns3d_cuda block3d_cuda)

if (INVIS_MODE)
  target_compile_definitions(cbns3d_cuda
                             PRIVATE IS_INVISCID
			     )
endif()

# Create a toggle option for building tests
option(BUILD_TESTS "Build unit tests" OFF)

# Add tests if the option is enabled
if(BUILD_TESTS)
  message(STATUS "Building unit tests")

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)  

  enable_testing()
  
  add_executable(unit_tests
                 test/gradient_test.cpp
  	         )
  
  target_link_libraries(unit_tests gtest_main block3d_cuda)
  
  include(GoogleTest)
  gtest_discover_tests(unit_tests)
else()
  message(STATUS "Unit tests are disabled")
endif()
