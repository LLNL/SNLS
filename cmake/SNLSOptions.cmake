option(USE_BATCH_SOLVERS "Build library with batch solvers which require the RAJA Peformance Suite" OFF)

option(USE_EXPT_RAJA "Make use of experimental RAJA features such as the dynamic forall feature.
                      Only useable when USE_BATCH_SOLVERS option is on." OFF)

option(USE_LAPACK "Build SNLS with LAPACK support" OFF)

option(ENABLE_TESTS "Enable tests" OFF)

option(ENABLE_GTEST "Enable gtest" OFF)

option(ENABLE_FRUIT "Enable fruit" OFF)

option(ENABLE_CUDA "Enable CUDA" OFF)

option(ENABLE_OPENMP "Enable openmp" OFF)