###############################################################################
#
# Setup UMPIRE
# This file defines:
#  UMPIRE_FOUND - If UMPIRE was found
#  UMPIRE_INCLUDE_DIRS - The UMPIRE include directories
#  UMPIRE_LIBRARY - The UMPIRE library

# first Check for UMPIRE_DIR

if(NOT UMPIRE_DIR)
    MESSAGE(FATAL_ERROR "Could not find UMPIRE. UMPIRE support needs explicit UMPIRE_DIR")
endif()

# umpire's installed cmake config target is lower case
set(umpire_DIR ${UMPIRE_DIR})
list(APPEND CMAKE_PREFIX_PATH ${umpire_DIR})

find_package(umpire REQUIRED)

set (UMPIRE_FOUND ${umpire_FOUND} CACHE STRING "")

set(UMPIRE_LIBRARIES umpire)

set(UMPIRE_DEPENDS camp)
blt_list_append(TO UMPIRE_DEPENDS ELEMENTS mpi IF ENABLE_MPI)
blt_list_append(TO UMPIRE_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)
blt_list_append(TO UMPIRE_DEPENDS ELEMENTS hip IF ENABLE_HIP)