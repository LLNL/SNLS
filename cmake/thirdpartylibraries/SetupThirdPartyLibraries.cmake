# Provide backwards compatibility for *_PREFIX options
set(_tpls 
    umpire
    raja)

foreach(_tpl ${_tpls})
    string(TOUPPER ${_tpl} _uctpl)
    if (${_uctpl}_PREFIX)
        set(${_uctpl}_DIR ${${_uctpl}_PREFIX} CACHE PATH "")
        mark_as_advanced(${_uctpl}_PREFIX)
    endif()
endforeach()

################################
# RAJA
################################

if (DEFINED RAJA_DIR)
    include(cmake/thirdpartylibraries/FindRAJA.cmake)
    if (RAJA_FOUND)
        blt_register_library( NAME       raja
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${RAJA_INCLUDE_DIRS}
                              LIBRARIES  ${RAJA_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find RAJA with given path ${RAJA_DIR}")
    endif()
else()
    message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
endif()


################################
# UMPIRE
################################

if (DEFINED UMPIRE_DIR)
    include(cmake/thirdpartylibraries/FindUmpire.cmake)
    if (RAJA_FOUND)
        blt_register_library( NAME       umpire
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${UMPIRE_INCLUDE_DIRS}
                              LIBRARIES  ${UMPIRE_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find UMPIRE with given path ${UMPIRE_DIR}")
    endif()
else()
    message(FATAL_ERROR "UMPIRE_DIR was not provided. It is needed to find UMPIRE.")
endif()
