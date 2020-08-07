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
if (NOT DEFINED camp_DIR)
   set(camp_DIR "${UMPIRE_DIR}/lib/cmake/camp")
endif()
find_package(camp)
find_package(umpire REQUIRED)

set (UMPIRE_FOUND ${umpire_FOUND} CACHE STRING "")
set (UMPIRE_LIBRARY umpire)
