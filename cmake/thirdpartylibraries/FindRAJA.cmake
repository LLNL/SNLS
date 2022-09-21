###############################################################################
#
# Setup RAJA
# This file defines:
#  RAJA_FOUND - If RAJA was found
#  RAJA_INCLUDE_DIRS - The RAJA include directories
#  RAJA_LIBRARY - The RAJA library

# first Check for RAJA_DIR

if(NOT RAJA_DIR)
    MESSAGE(FATAL_ERROR "Could not find RAJA. RAJA support needs explicit RAJA_DIR")
endif()

set(raja_DIR ${RAJA_DIR})

if (NOT RAJA_CONFIG_CMAKE)
   set(RAJA_CONFIG_CMAKE "${RAJA_DIR}/share/raja/cmake/raja-config.cmake")
endif()
if (EXISTS "${RAJA_CONFIG_CMAKE}")
   include("${RAJA_CONFIG_CMAKE}")
endif()
if (NOT RAJA_RELEASE_CMAKE)
   set(RAJA_RELEASE_CMAKE "${RAJA_DIR}/share/raja/cmake/raja-release.cmake")
endif()
if (EXISTS "${RAJA_RELEASE_CMAKE}")
   include("${RAJA_RELEASE_CMAKE}")
endif()

find_package(raja REQUIRED)

get_target_property(RAJA_INCLUDE_DIRS RAJA INTERFACE_INCLUDE_DIRECTORIES)
set(RAJA_LIBRARIES RAJA)
set(RAJA_DEPENDS camp)
blt_list_append(TO RAJA_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)
blt_list_append(TO RAJA_DEPENDS ELEMENTS openmp IF ENABLE_OPENMP)
blt_list_append(TO RAJA_DEPENDS ELEMENTS hip IF ENABLE_HIP)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RAJA  DEFAULT_MSG
                                  RAJA_INCLUDE_DIRS
                                  RAJA_LIBRARIES )