include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(WARNING "spmm: Cannot find CUDA, turn off the `USE_CUDA` option automatically.")
  set(USE_CUDA OFF)
  return()
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDA QUIET REQUIRED)

message(${CUDA_INCLUDE_DIRS})
message(${CUDA_TOOLKIT_ROOT_DIR})
add_library(spmm::cudart INTERFACE IMPORTED)
set_property(TARGET spmm::cudart PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES})
set_property(TARGET spmm::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

# cublas
add_library(spmm::cublas INTERFACE IMPORTED)
message(${CUDA_CUBLAS_LIBRARIES})
if(BUILD_SHARED_LIBS)
    set_property(TARGET spmm::cublas PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
else()
    set_property(TARGET spmm::cublas PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
endif()
set_property(TARGET spmm::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

# cusparse
add_library(spmm::cusparse INTERFACE IMPORTED)
message(${CUDA_cusparse_LIBRARY})
if(BUILD_SHARED_LIBS)
    set_property(TARGET spmm::cusparse PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_cusparse_LIBRARY})
else()
    set_property(TARGET spmm::cusparse PROPERTY INTERFACE_LINK_LIBRARIES
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse_static.a")
endif()
set_property(TARGET spmm::cusparse PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

# Thrust is a Head-only library
add_library(spmm::thrust INTERFACE IMPORTED)
set_property(TARGET spmm::thrust PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-declarations")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
# list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70")
# list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_80,code=sm_80")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_90,code=sm_90")
STRING(REPLACE ";" " " CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
list(APPEND CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})
message(STATUS "Add CUDA NVCC flags: ${CUDA_NVCC_FLAGS}")