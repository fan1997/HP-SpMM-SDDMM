cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(spmm VERSION 0.0.1 LANGUAGES CXX)

set(USE_CUDA ON CACHE BOOL "Compile with CUDA")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Compiled as a shared library")
set(ENABLE_TESTING ON CACHE BOOL "Build with testing enabled")
set(USE_CUDA ON CACHE BOOL "Build with CUDA supported")
set(USE_CUSP ON CACHE BOOL "Build with CUSP supported")
set(USE_STARML ON CACHE BOOL "Build with starml supported")
set(USE_CUDA11 ON CACHE BOOL "Build with CUDA11 supported")


SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
message(${CMAKE_CXX_FLAGS_RELEASE})

# Show the detail of compile messages
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_COLOR_MAKEFILE   ON)
include(cmake/public/Utils.cmake)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(EXTERNAL_DIR ${AITISA_API_ROOT_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Find all the files under `spmm/`
file(GLOB_RECURSE srcs "spmm/*.cpp" "spmm/*.cu")
if(NOT USE_CUDA)
  file(GLOB_RECURSE cuda_srcs "spmm/*_cuda.cpp" "spmm/*.cu")
  exclude(srcs "${srcs}" ${cuda_srcs})
endif()

# Create library target with all the source codes.
if(BUILD_SHARED_LIBS)
  add_library(spmm SHARED ${srcs})
else()
  add_library(spmm STATIC ${srcs})
endif()

target_include_directories(spmm
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
if(USE_CUDA)
  set_target_properties(spmm PROPERTIES 
    CUDA_SEPERABLE_COMPILATION ON 
    INTERFACE_POSITION_INDEPENDENT_CODE ON 
  )
  set_target_properties(spmm PROPERTIES CUDA_ARCHITECTURES "80")
endif()

# Add dependency library link
set(link_libraries)
#for saving third_party source 
set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)

if(USE_CUDA)
  include(cmake/public/CUDA.cmake)
  list(APPEND link_libraries spmm::cusparse spmm::cudart spmm::cublas spmm::thrust)
endif()

find_package(OpenMP REQUIRED QUIET)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

if(USE_CUSP)
  include(cmake/public/cusp.cmake)
  list(APPEND link_libraries spmm::cusp)
endif()

target_link_libraries(spmm PUBLIC ${link_libraries})

# Add testing
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

