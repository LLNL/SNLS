blt_add_executable(NAME       snlstest
                   SOURCES    SNLS_testdriver.cc
                   HEADERS    SNLS_testmodels.h
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls)
blt_add_test(NAME    snlstest
             COMMAND snlstest)

blt_add_executable(NAME       snlsNLS
                   SOURCES    SNLS_testdriver_hybrd.cc
                   HEADERS    SNLS_testmodels.h
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls)

blt_add_test(NAME    snlsNLS
             COMMAND snlsNLS)

set(SNLS_TEST_DEPENDS)

snls_fill_depends_list(LIST_NAME  SNLS_TEST_DEPENDS
                       DEPENDS_ON chai RAJA umpire camp)

if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
  if(ENABLE_CUDA)
    list(APPEND SNLS_TEST_DEPENDS blt::cuda_runtime blt::cuda)
  endif()
  if(ENABLE_OPENMP)
    list(APPEND SNLS_TEST_DEPENDS blt::openmp)
  endif()
else()
  if(ENABLE_CUDA)
    list(APPEND SNLS_TEST_DEPENDS cuda cuda_runtime) 
  endif()
  if(ENABLE_OPENMP)
    list(APPEND SNLS_TEST_DEPENDS openmp)
  endif()
endif()

if(ENABLE_HIP)
  list(APPEND SNLS_TEST_DEPENDS blt::hip blt::hip_runtime)
endif()

blt_add_executable(NAME       snls_forall_subviews
                  SOURCES    SNLS_forall_subviews.cxx
                  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                  DEPENDS_ON gtest snls ${SNLS_TEST_DEPENDS})
blt_add_test(NAME    snls_forall_subviews
             COMMAND snls_forall_subviews)

if(USE_BATCH_SOLVERS)

  blt_add_executable(NAME       snls_mm
                    SOURCES    SNLS_mm.cxx
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest snls ${SNLS_TEST_DEPENDS})
  blt_add_test(NAME    snls_mm
              COMMAND snls_mm)

  blt_add_executable(NAME       snlstest_batch
                    SOURCES    SNLS_batch_testdriver.cc
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest snls ${SNLS_TEST_DEPENDS})
  blt_add_test(NAME    snlstest_batch
              COMMAND snlstest_batch)

  blt_add_executable(NAME snls_view_tests
                    SOURCES    SNLS_view_tests.cc
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest snls ${SNLS_TEST_DEPENDS})

  blt_add_test(NAME    snls_view_tests
              COMMAND snls_view_tests)

endif()

if(ENABLE_BENCHMARKS)
  blt_add_executable(NAME    snls_benchmark
                     SOURCES    SNLS_benchmark.cc
                     HEADERS    SNLS_testmodels.h
                     OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                     DEPENDS_ON gbenchmark snls ${SNLS_TEST_DEPENDS})

  blt_add_benchmark(NAME    snls_benchmark
                    COMMAND snls_benchmark --benchmark_time_unit=us)
endif(ENABLE_BENCHMARKS)