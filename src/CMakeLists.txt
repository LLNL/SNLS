set(SNLS_HEADERS
    ${SNLS_HEADER_INCLUDE_DIR}/SNLS_config.h
    SNLS_base.h
    SNLS_gpu_portability.h
    SNLS_port.h
    SNLS_unused.h
    SNLS_lup_solve.h
    SNLS_TrDelta.h
    SNLS_HybrdTrDLDenseG.h
    SNLS_TrDLDenseG.h
    SNLS_TrDLDenseG_Batch.h
    SNLS_NewtonBB.h
    SNLS_view_types.h
    SNLS_device_forall.h
    SNLS_memory_manager.h
    SNLS_qr_solve.h
    SNLS_linalg.h
    SNLS_kernels.h
    SNLS_kernels_batch.h)

set(SNLS_SOURCES
    SNLS_device_forall.cxx
    SNLS_memory_manager.cxx)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(SNLS_DEPENDS)

snls_fill_depends_list(LIST_NAME  SNLS_DEPENDS
                       DEPENDS_ON chai RAJA umpire camp)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    if(ENABLE_CUDA)
        list(APPEND SNLS_DEPENDS blt::cuda_runtime blt::cuda)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND SNLS_DEPENDS blt::openmp)
    endif()
else()
    if(ENABLE_CUDA)
        list(APPEND SNLS_DEPENDS cuda cuda_runtime)
    endif()
    if(ENABLE_OPENMP)
        list(APPEND SNLS_DEPENDS openmp)
    endif()
endif()

if(ENABLE_HIP)
    list(APPEND SNLS_DEPENDS blt::hip blt::hip_runtime)
endif()

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
