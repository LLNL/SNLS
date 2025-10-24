#
# see README.md for copyright and license information
# 

cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0057 NEW)

if (APPLE)
    cmake_policy(SET CMP0025 NEW)
endif()

message(STATUS "Using cmake version ${CMAKE_VERSION}")

project(SNLS LANGUAGES CXX)

# GTest requires a C compiler when running with HIP...
# On other systems this isn't necessarily an issue as CMake will
# just pick-up a valid system C compiler but on our AMD GPU machines
# that doesn't always happen.
if (ENABLE_HIP AND ENABLE_TESTS)
    enable_language(C)
endif()

set(CMAKE_CXX_STANDARD 17)

#--------------------------------------------------------------------------------
# BLT
#--------------------------------------------------------------------------------
set(BLT_CXX_STD "c++17" CACHE STRING "")

if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given

    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT submodule is not present. "
            "Run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update" )
    endif()
endif()

set(ENABLE_GTEST OFF CACHE BOOL "")
set(ENABLE_FRUIT OFF CACHE BOOL "")
set(ENABLE_GBENCHMARK OFF CACHE BOOL "")

if(ENABLE_TESTS)
    set(ENABLE_GTEST ON CACHE BOOL "" FORCE)
endif(ENABLE_TESTS)

if(ENABLE_BENCHMARKS)
    set(ENABLE_BENCHMARKS ON CACHE BOOL "" FORCE)
    set(ENABLE_GBENCHMARK ON CACHE BOOL "" FORCE)
endif(ENABLE_BENCHMARKS)

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    blt_install_tpl_setups(DESTINATION share/snls/cmake/)
endif()
# turn off testing if building under MSLib, so that do not have to link ms to have MS_Fail
if (DEFINED MSLIB_SOURCE_DIR)
    set(ENABLE_TESTS OFF)
endif (DEFINED MSLIB_SOURCE_DIR)

# turn off testing if building under ECMech, because gtest dependence does not seem to work
if (DEFINED ECMECH_SOURCE_DIR)
    set(ENABLE_TESTS OFF)
endif (DEFINED ECMECH_SOURCE_DIR)
################################
# Include standard build system logic and options
################################
include(cmake/CMakeBasics.cmake)

if(ENABLE_CUDA)
    if (NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-restrict")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict")
    endif()
    # Can be --expt-extended-lambda or --extended-lambda (newer CUDA versions only)
    if (NOT "${CMAKE_CUDA_FLAGS}" MATCHES "extended-lambda")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    endif()

    if (NOT "${CMAKE_CUDA_FLAGS}" MATCHES "--expt-relaxed-constexpr")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    endif()
endif()

#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
add_subdirectory(src)

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

if(ENABLE_TESTS)
   add_subdirectory(test)
endif(ENABLE_TESTS)

