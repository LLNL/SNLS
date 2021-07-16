# Provide backwards compatibility for *_PREFIX options
set(_tpls
    camp
    raja
    umpire
    chai)

foreach(_tpl ${_tpls})
    string(TOUPPER ${_tpl} _uctpl)
    if (${_uctpl}_PREFIX)
        set(${_uctpl}_DIR ${${_uctpl}_PREFIX} CACHE PATH "")
        mark_as_advanced(${_uctpl}_PREFIX)
    endif()
endforeach()

################################
# camp
################################

if (CAMP_DIR)
    include(cmake/thirdpartylibraries/Findcamp.cmake)
    if (CAMP_FOUND)
        blt_register_library( NAME       camp
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${CAMP_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Unable to find camp with given path ${CAMP_DIR}")
    endif()
endif()

################################
# RAJA
################################

if (DEFINED RAJA_DIR)
    include(cmake/thirdpartylibraries/FindRAJA.cmake)
    if (RAJA_FOUND)
        blt_register_library( NAME       raja
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${RAJA_INCLUDE_DIRS}
                              DEPENDS_ON ${RAJA_DEPENDS}
                              LIBRARIES  ${RAJA_LIBRARIES}
                              DEFINES    HAVE_RAJA)
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
    if (UMPIRE_FOUND)
        blt_register_library( NAME       umpire
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${UMPIRE_INCLUDE_DIRS}
                              DEPENDS_ON ${UMPIRE_DEPENDS}
                              LIBRARIES  ${UMPIRE_LIBRARIES}
                              DEFINES    HAVE_UMPIRE)
    else()
        message(FATAL_ERROR "Unable to find UMPIRE with given path ${UMPIRE_DIR}")
    endif()
else()
    message(FATAL_ERROR "UMPIRE_DIR was not provided. It is needed to find UMPIRE.")
endif()

################################
# CHAI
################################

if (DEFINED CHAI_DIR)
    include(cmake/thirdpartylibraries/FindCHAI.cmake)
    if (CHAI_FOUND)
        blt_register_library( NAME       chai
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${CHAI_INCLUDE_DIRS}
                              DEPENDS_ON ${CHAI_DEPENDS}
                              LIBRARIES  ${CHAI_LIBRARIES}
                              DEFINES    HAVE_CHAI)
    else()
        message(FATAL_ERROR "Unable to find CHAI with given path ${CHAI_DIR}")
    endif()
else()
    message(FATAL_ERROR "CHAI_DIR was not provided. It is needed to find CHAI.")
endif()