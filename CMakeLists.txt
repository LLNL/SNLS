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

#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(SNLS_HEADERS
    SNLS_base.h
    SNLS_cuda_portability.h
    SNLS_port.h
    SNLS_lup_solve.h
    SNLS_TrDLDenseG.h
    SNLS_NewtonBB.h)

# no SNLS_SOURCES -- SNLS is header-only

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------

SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(SNLS_INCLUDES ${HEADER_INCLUDE_DIR}
                  ${CMAKE_SOURCE_DIR}/snls)
if (DEFINED MSLIB_SOURCE_DIR)
    list(APPEND SNLS_INCLUDES ${MSLIB_SOURCE_DIR}/ms)
endif (DEFINED MSLIB_SOURCE_DIR)

#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------
blt_add_library(NAME        snls
                INCLUDES    ${SNLS_INCLUDES}
                HEADERS     ${SNLS_HEADERS})

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

if(ENABLE_TESTS)
    # do not have DEPENDS_ON snls, given that it is header-only 

    blt_add_executable(NAME       snlstest
                       SOURCES    SNLS_testdriver.cc
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON gtest)
    blt_add_test(NAME    snlstest
                 COMMAND snlstest)
endif(ENABLE_TESTS)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

install(FILES ${SNLS_HEADERS} DESTINATION include)

install(TARGETS snls DESTINATION lib)
