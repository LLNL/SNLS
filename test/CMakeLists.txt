# do not have DEPENDS_ON snls, given that it is header-only 

blt_add_executable(NAME       snlstest
                   SOURCES    SNLS_testdriver.cc
                   OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                   DEPENDS_ON gtest snls)
blt_add_test(NAME    snlstest
             COMMAND snlstest)
