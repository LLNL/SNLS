set(SNLS_HEADERS
    ${SNLS_HEADER_INCLUDE_DIR}/SNLS_config.h
    SNLS_base.h
    SNLS_cuda_portability.h
    SNLS_port.h
    SNLS_lup_solve.h
    SNLS_TrDelta.h
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
                       DEPENDS_ON chai raja umpire camp openmp cuda_runtime cuda)

message("-- SNLS_DEPENDS: ${SNLS_DEPENDS}")

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------
blt_add_library(NAME        snls
                OUTPUT_NAME snls
                DEFINES     ${SNLS_DEFINES}
                DEPENDS_ON  ${SNLS_DEPENDS}
                HEADERS     ${SNLS_HEADERS}
                SOURCES     ${SNLS_SOURCES}
                SHARED      FALSE)
#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

install(FILES ${SNLS_HEADERS} DESTINATION include/snls/)

target_include_directories(
  snls
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${SNLS_HEADER_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/SNLS-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/snls/cmake/snls-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/snls/cmake/snls-config.cmake
  DESTINATION share/snls/cmake/)

install(
  FILES ${SNLS_HEADER_INCLUDE_DIR}/SNLS_config.h
  DESTINATION include/snls/)

install(
  FILES ${SNLS_HEADERS}
  DESTINATION include/snls/)

install(
  TARGETS snls
  EXPORT snls-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(EXPORT snls-targets DESTINATION share/snls/cmake/)