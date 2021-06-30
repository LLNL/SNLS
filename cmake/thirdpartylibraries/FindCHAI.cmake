###############################################################################
#
# Setup CHAI
# This file defines:
#  CHAI_FOUND - If CHAI was found
#  CHAI_INCLUDE_DIRS - The CHAI include directories
#  CHAI_LIBRARY - The CHAI library

# first Check for CHAI_DIR

if(NOT CHAI_DIR)
    MESSAGE(FATAL_ERROR "Could not find CHAI. CHAI support needs explicit CHAI_DIR")
endif()

# chai's installed cmake config target is lower case
set(chai_DIR ${CHAI_DIR})
list(APPEND CMAKE_PREFIX_PATH ${chai_DIR})

find_package(chai REQUIRED)

set (CHAI_FOUND ${chai_FOUND} CACHE STRING "")

set(CHAI_LIBRARIES chai)

set(UMPIRE_DEPENDS camp umpire raja)
blt_list_append(TO CHAI_DEPENDS ELEMENTS mpi IF ENABLE_MPI)
blt_list_append(TO CHAI_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)
blt_list_append(TO CHAI_DEPENDS ELEMENTS hip IF ENABLE_HIP)