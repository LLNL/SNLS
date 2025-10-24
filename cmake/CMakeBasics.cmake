################################
# Version
################################
set(PACKAGE_BUGREPORT "carson16@llnl.gov")

set(SNLS_VERSION_MAJOR 0)
set(SNLS_VERSION_MINOR 4)
set(SNLS_VERSION_PATCH \"2\")

set(SNLS_HEADER_INCLUDE_DIR
    ${PROJECT_BINARY_DIR}/include/snls
    CACHE PATH
    "Directory where all generated headers will go in the build tree")

################################
# Setup build options and their default values
################################
#include(cmake/SNLSOptions.cmake)

##############################
# settings into SNLS_config.h
##############################

set(HAVE_SNLS "1" CACHE STRING "")

if(USE_RAJA_ONLY)
    set(SNLS_RAJA_ONLY "1" CACHE STRING "")
    set(SNLS_USE_RAJA_ONLY TRUE CACHE BOOL "")
elseif(USE_BATCH_SOLVERS)
    set(SNLS_RAJA_PORT_SUITE "1" CACHE STRING "")
    set(SNLS_USE_RAJA_PORT_SUITE TRUE CACHE BOOL "")
endif()

if(USE_LAPACK)
    set(SNLS_USE_LAPACK "1" CACHE STRING "")
endif()


if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(SNLS_DEBUG "1" CACHE STRING "")
endif()

configure_file( src/SNLS_config.h.in
                ${SNLS_HEADER_INCLUDE_DIR}/SNLS_config.h )

################################
# Third party library setup
################################
include(cmake/thirdpartylibraries/SetupThirdPartyLibraries.cmake)

##------------------------------------------------------------------------------
## snls_fill_depends_on_list
##
## This macro adds a dependency to the list if ENABLE_<dep name> or <dep name>_FOUND
##------------------------------------------------------------------------------
macro(snls_fill_depends_list)

    set(options )
    set(singleValueArgs LIST_NAME)
    set(multiValueArgs DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT arg_LIST_NAME)
        message(FATAL_ERROR "snls_fill_depends_list requires argument LIST_NAME")
    endif()

    if (NOT arg_DEPENDS_ON)
        message(FATAL_ERROR "snls_fill_depends_list requires argument DEPENDS_ON")
    endif()

    foreach( _dep ${arg_DEPENDS_ON})
        string(TOUPPER ${_dep} _ucdep)

        if (ENABLE_${_ucdep} OR ${_ucdep}_FOUND OR ${_dep}_FOUND)
            list(APPEND ${arg_LIST_NAME} ${_dep})
        endif()
    endforeach()

    unset(_dep)
    unset(_ucdep)

endmacro(snls_fill_depends_list)
