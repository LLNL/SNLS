blt_add_executable(NAME       snlstest
                   SOURCES    SNLS_testdriver.cc
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls)
blt_add_test(NAME    snlstest
             COMMAND snlstest)

blt_add_executable(NAME       snlsNLS
                   SOURCES    SNLS_testdriver_hybrd.cc
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls)

blt_add_test(NAME    snlsNLS
             COMMAND snlsNLS)

set(SNLS_TEST_DEPENDS)

snls_fill_depends_list(LIST_NAME  SNLS_TEST_DEPENDS
                       DEPENDS_ON chai raja umpire camp openmp cuda_runtime cuda blt::hip blt::hip_runtime)

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
