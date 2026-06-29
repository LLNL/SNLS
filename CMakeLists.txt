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

project(SNLS
        LANGUAGES CXX
        VERSION 0.4.4)

# GTest requires a C compiler when running with HIP...
# On other systems this isn't necessarily an issue as CMake will
# just pick-up a valid system C compiler but on our AMD GPU machines
# that doesn't always happen.
if (ENABLE_HIP AND ENABLE_TESTS)
    enable_language(C)
endif()

include(cmake/SetupBLTWrapper.cmake)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_RAJA_ONLY "Use RAJA" OFF)
option(USE_BATCH_SOLVERS "Use batch solvers" OFF)

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

# SNLS_ENABLE_TESTS allows downstream projects to enable BLT's unit testing
# framework separately from SNLS's tests.
include(CMakeDependentOption)
cmake_dependent_option(SNLS_ENABLE_TESTS "Enable SNLS tests"
                       ON "ENABLE_TESTS" OFF)

if(SNLS_ENABLE_TESTS)
  add_subdirectory(test)
endif()
