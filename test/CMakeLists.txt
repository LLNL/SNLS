# do not have DEPENDS_ON snls, given that it is header-only 

blt_add_executable(NAME       snlstest
                   SOURCES    SNLS_testdriver.cc
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls)
blt_add_test(NAME    snlstest
             COMMAND snlstest)
set(SNLS_DEPENDS)

snls_fill_depends_list(LIST_NAME  SNLS_DEPENDS
                       DEPENDS_ON raja umpire chai openmp cuda_runtime cuda)

blt_add_executable(NAME       snls_mm
                   SOURCES    SNLS_mm.cxx
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls ${SNLS_DEPENDS})
blt_add_test(NAME    snls_mm
             COMMAND snls_mm)

blt_add_executable(NAME       snlstest_batch
                   SOURCES    SNLS_batch_testdriver.cc
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls ${SNLS_DEPENDS})
blt_add_test(NAME    snlstest_batch
             COMMAND snlstest_batch)

blt_add_executable(NAME       chai_exp
                   SOURCES    chai_min_exp.cxx
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest ${SNLS_DEPENDS})
blt_add_test(NAME    chai_exp
             COMMAND chai_exp)
