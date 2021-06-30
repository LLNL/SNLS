######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup CUB
# This file defines:
#  CUB_FOUND - If CUB was found
#  CUB_INCLUDE_DIRS - The CUB include directories
#  cub - An imported target

# Borrowed this logic from CARE's FindCUB.cmake
if (NOT TARGET cub)
   if (CUB_DIR)
      set(CUB_PATHS ${CUB_DIR} ${CUB_DIR}/include)
   endif ()

   find_path(CUB_INCLUDE_DIR
             NAMES cub/cub.cuh
             PATHS ${CUB_PATHS}
             NO_DEFAULT_PATH
             NO_PACKAGE_ROOT_PATH
             NO_CMAKE_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

   include(FindPackageHandleStandardArgs)

   find_package_handle_standard_args(CUB
                                     DEFAULT_MSG
                                     CUB_INCLUDE_DIR)

   if (CUB_FOUND)
      set(CUB_INCLUDE_DIRS ${CUB_INCLUDE_DIR})
      set(CUB_DEPENDS cuda)

      blt_register_library(NAME cub
                           INCLUDES ${CUB_INCLUDE_DIR}
                           DEPENDS_ON ${CUB_DEPENDS})
      message(STATUS "SNSL: CUB found at ${CUB_INCLUDE_DIR}")
   else ()
      message(FATAL_ERROR "SNLS: CUB not found. Set CUB_DIR to use an external build of CUB.")
   endif ()
endif ()