set(SNLS_HEADERS
    SNLS_base.h
    SNLS_cuda_portability.h
    SNLS_port.h
    SNLS_lup_solve.h
    SNLS_TrDLDenseG.h
    SNLS_TrDLDenseG_Batch.h
    SNLS_NewtonBB.h
    SNLS_device_forall.h
    SNLS_memory_manager.h)

set(SNLS_SOURCES
    SNLS_device_forall.cxx
    SNLS_memory_manager.cxx)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(SNLS_DEPENDS)

snls_fill_depends_list(LIST_NAME  SNLS_DEPENDS
                       DEPENDS_ON raja umpire cuda openmp)

message("-- SNLS_DEPENDS: ${SNLS_DEPENDS}")

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
                DEPENDS_ON  ${SNLS_DEPENDS}
                INCLUDES    ${SNLS_INCLUDES}
                HEADERS     ${SNLS_HEADERS}
                SOURCES     ${SNLS_SOURCES})
#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

install(FILES ${SNLS_HEADERS} DESTINATION include)

install(TARGETS snls DESTINATION lib)
