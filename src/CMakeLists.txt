set(SNLS_HEADERS
    SNLS_base.h
    SNLS_cuda_portability.h
    SNLS_port.h
    SNLS_lup_solve.h
    SNLS_TrDLDenseG.h
    SNLS_NewtonBB.h)
# no SNLS_SOURCES -- SNLS is header-only

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------

SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(SNLS_INCLUDES ${HEADER_INCLUDE_DIR}
                  ${CMAKE_SOURCE_DIR}/snls/src)
if (DEFINED MSLIB_SOURCE_DIR)
    list(APPEND SNLS_INCLUDES ${MSLIB_SOURCE_DIR}/ms)
endif (DEFINED MSLIB_SOURCE_DIR)

#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------
blt_add_library(NAME        snls
                INCLUDES    ${SNLS_INCLUDES}
                HEADERS     ${SNLS_HEADERS})
#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

install(FILES ${SNLS_HEADERS} DESTINATION include)

install(TARGETS snls DESTINATION lib)
