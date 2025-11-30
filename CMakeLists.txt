cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(mganns
  VERSION 0.0.1
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL ""
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

message(STATUS "Environment variables (PATH):" $ENV{PATH})

# default to release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message (STATUS "Build type: ${CMAKE_BUILD_TYPE}")

SET(MKL_INTERFACE_FULL intel_lp64)
find_package(MKL REQUIRED)
message("-------- MKL OK -------- ")
find_package(OpenMP REQUIRED)
message("-------- OpenMP  OK -------- ")
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt -pthread -march=native -mfma -ffp-contract=fast -ffast-math -ftree-vectorize -ftree-vectorizer-verbose=0")
message("!! : CMAKE_CXX_FLAGS now is ${CMAKE_CXX_FLAGS}")

# set (LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/")
# set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/${CMAKE_BUILD_TYPE}/")
# set (LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
# set (EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

include_directories(thirdparty/hnswlib)
include_directories(thirdparty/gp-ann/src)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/rapidjson/include)

add_subdirectory(thirdparty/gp-ann ${CMAKE_CURRENT_BINARY_DIR}/gp-ann_build)

include_directories(src)

add_subdirectory(src)

file(COPY scripts/run.sh DESTINATION ${CMAKE_BINARY_DIR})
execute_process(
  COMMAND chmod +x ${CMAKE_BINARY_DIR}/run.sh
)