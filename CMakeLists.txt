#
# see README.md for copyright and license information
# 

cmake_minimum_required(VERSION 3.1)
if("${CMAKE_VERSION}" VERSION_GREATER 3.3)
    cmake_policy(SET CMP0057 NEW)
endif()

message(STATUS "Using cmake version ${CMAKE_VERSION}")

project(SNLS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

#--------------------------------------------------------------------------------
# BLT
#--------------------------------------------------------------------------------
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

if(ENABLE_TESTS)
    set(ENABLE_GTEST ON CACHE BOOL "" FORCE)
endif(ENABLE_TESTS)

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

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
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict --expt-extended-lambda --expt-relaxed-constexpr")
  if (NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-arch")
     if (DEFINED CUDA_ARCH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch ${CUDA_ARCH}")
     endif()
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

