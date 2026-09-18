cmake_minimum_required(VERSION 3.24)
project(spark_kernels LANGUAGES CXX CUDA)

# Primary target: RTX 5090, compute capability 12.0 (sm_120). The DGX Spark (GB10) is 12.1:
# pass -DCMAKE_CUDA_ARCHITECTURES=121, or "120;121" for one build that runs on both.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit 13.0 REQUIRED)

option(SPARK_BUILD_BENCH "Build benchmark binaries (needs cuBLAS)" ON)
option(SPARK_LINEINFO "Embed line info for Nsight Compute source view" ON)

# ---- kernel library ----------------------------------------------------------------------
file(GLOB SPARK_KERNEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/*.cu)
add_library(spark_kernels STATIC ${SPARK_KERNEL_SOURCES})
target_include_directories(spark_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(spark_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${SPARK_LINEINFO}>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
set_target_properties(spark_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

# ---- benchmark binaries ------------------------------------------------------------------
if(SPARK_BUILD_BENCH)
  file(GLOB SPARK_BENCH_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/bench/bench_*.cu)
  foreach(src ${SPARK_BENCH_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src/bench)
    target_link_libraries(${name} PRIVATE spark_kernels CUDA::cublas CUDA::cudart)
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${SPARK_LINEINFO}>>:-lineinfo>)
  endforeach()
endif()
