###############################################################################
#
# Setup CAMP
# This file defines:
#  CAMP_FOUND - If CAMP was found
#  CAMP_INCLUDE_DIRS - The CAMP include directories

# first Check for CAMP_DIR

if(NOT CAMP_DIR)
    MESSAGE(FATAL_ERROR "Could not find CAMP. CAMP support needs explicit CAMP_DIR")
endif()

# camp's installed cmake config target is lower case
set(camp_DIR ${CAMP_DIR})
list(APPEND CMAKE_PREFIX_PATH ${camp_DIR})
find_package(camp REQUIRED)

set(CAMP_FOUND ${camp_FOUND} CACHE BOOL "Whether or not CAMP was found")
set(CAMP_INCLUDE_DIRS ${camp_INSTALL_PREFIX}/include CACHE PATH "CAMP include directories")