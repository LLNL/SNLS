set(SNLS_HEADERS
    ${HEADER_INCLUDE_DIR}/SNLS_config.h
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
                       DEPENDS_ON raja umpire chai openmp cuda_runtime cuda)

message("-- SNLS_DEPENDS: ${SNLS_DEPENDS}")

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(SNLS_INCLUDES ${HEADER_INCLUDE_DIR}
                  ${CMAKE_SOURCE_DIR}/src)
if (DEFINED MSLIB_SOURCE_DIR)
    list(APPEND SNLS_INCLUDES ${MSLIB_SOURCE_DIR}/ms)
endif (DEFINED MSLIB_SOURCE_DIR)

#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------
blt_add_library(NAME        snls
                OUTPUT_NAME snls
                DEFINES     ${SNLS_DEFINES}
                DEPENDS_ON  ${SNLS_DEPENDS}
                INCLUDES    ${SNLS_INCLUDES}
                HEADERS     ${SNLS_HEADERS}
                SOURCES     ${SNLS_SOURCES}
                SHARED      FALSE)
#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

set(SNLS_INSTALLED_HEADERS ${SNLS_HEADERS})

install(FILES ${SNLS_INSTALLED_HEADERS} DESTINATION include)

install(TARGETS snls DESTINATION lib)
