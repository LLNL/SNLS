#
# see README.md for copyright and license information
# 

cmake_minimum_required(VERSION 3.1)
if("${CMAKE_VERSION}" VERSION_GREATER 3.3)
    cmake_policy(SET CMP0057 NEW)
endif()

message(STATUS "Using cmake version ${CMAKE_VERSION}")

project(SNLS LANGUAGES CXX)

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

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

# turn off testing if building under MSLib, so that do not have to link ms to have MS_Fail
if(ENABLE_MSLIB_BUILD)
    set(ENABLE_TESTS OFF)
endif(ENABLE_MSLIB_BUILD)

#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(SNLS_HEADERS
    SNLS_port.h
    SNLS_lup_solve.h
    SNLS_TrDLDenseG.h)

set(SNLS_SOURCES
    SNLS_lup_solve.cc
    SNLS_TrDLDenseG.cxx)

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------
set(SNLS_DEFINES)

if(ENABLE_MSLIB_BUILD)
    list(APPEND SNLS_DEFINES HAVE_MSLIB)
endif(ENABLE_MSLIB_BUILD)

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(SNLS_INCLUDES ${HEADER_INCLUDE_DIR}
                  ${CMAKE_SOURCE_DIR}/snls)
if(ENABLE_MSLIB_BUILD)
    list(APPEND SNLS_INCLUDES ${CMAKE_SOURCE_DIR}/ms)
endif(ENABLE_MSLIB_BUILD)

#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------
if(BUILD_SHARED_LIBS)
    blt_add_library(NAME        snls_shared
                    OUTPUT_NAME snls
                    DEFINES     ${SNLS_DEFINES}
                    INCLUDES    ${SNLS_INCLUDES}
                    DEPENDS_ON  ${SNLS_DEPENDS}
                    HEADERS     ${SNLS_HEADERS}
                    SOURCES     ${SNLS_SOURCES}
                    SHARED      TRUE)
endif(BUILD_SHARED_LIBS)

if(BUILD_STATIC_LIBS)
    blt_add_library(NAME        snls_static
                    OUTPUT_NAME snls
                    DEFINES     ${SNLS_DEFINES}
                    INCLUDES    ${SNLS_INCLUDES}
                    DEPENDS_ON  ${SNLS_DEPENDS}
                    HEADERS     ${SNLS_HEADERS}
                    SOURCES     ${SNLS_SOURCES}
                    SHARED      FALSE)
endif(BUILD_STATIC_LIBS)

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------

if(ENABLE_TESTS)
    if (BUILD_STATIC_LIBS)
        set(SNLSTEST_DEPENDS snls_static)
    else()
        set(SNLSTEST_DEPENDS snls_shared)
    endif()

    blt_add_executable(NAME       snlstest
                       SOURCES    SNLS_testdriver.cc
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                       DEPENDS_ON ${SNLSTEST_DEPENDS})
    blt_add_test(NAME    snlstest
                 COMMAND snlstest)
endif(ENABLE_TESTS)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

set(SNLS_INSTALLED_HEADERS
    SNLS_lup_solve.h
    SNLS_TrDLDenseG.h)

install(FILES ${SNLS_INSTALLED_HEADERS} DESTINATION include)

if(BUILD_SHARED_LIBS)
    install(TARGETS snls_shared DESTINATION lib)
endif(BUILD_SHARED_LIBS)

if(BUILD_STATIC_LIBS)
    install(TARGETS snls_static DESTINATION lib)
endif(BUILD_STATIC_LIBS)
