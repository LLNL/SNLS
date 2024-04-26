# Provide backwards compatibility for *_PREFIX options
set(_tpls
    camp
    raja
    umpire
    chai)

foreach(_tpl ${_tpls})
    string(TOUPPER ${_tpl} _uctpl)
    if (${_uctpl}_PREFIX)
        set(${_uctpl}_DIR ${${_uctpl}_PREFIX} CACHE PATH "")
        mark_as_advanced(${_uctpl}_PREFIX)
    endif()
endforeach()

# Only search for these if the batch solver is enabled
if(USE_BATCH_SOLVERS)

message("Use SNLS batch solvers...")

################################
# camp
################################

if (CAMP_DIR)
   find_package(camp REQUIRED CONFIG PATHS ${CAMP_DIR})
   find_package_handle_standard_args(camp REQUIRED)
else()
    message(FATAL_ERROR "CAMP_DIR was not provided. It is needed to find CAMP.")
endif()


################################
# chai
################################

if (DEFINED CHAI_DIR)
   find_package(chai REQUIRED CONFIG PATHS ${CHAI_DIR})
   find_package_handle_standard_args(chai REQUIRED)
else()
    message(FATAL_ERROR "CHAI_DIR was not provided. It is needed to find CHAI.")
endif()

################################
# RAJA
################################

if (DEFINED RAJA_DIR)
   find_package(RAJA REQUIRED CONFIG PATHS ${RAJA_DIR})
   find_package_handle_standard_args(RAJA REQUIRED)
else()
    message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
endif()


################################
# UMPIRE
################################

if (DEFINED UMPIRE_DIR)
   find_package(umpire REQUIRED CONFIG PATHS ${UMPIRE_DIR})
   find_package_handle_standard_args(umpire REQUIRED)
else()
    message(FATAL_ERROR "UMPIRE_DIR was not provided. It is needed to find UMPIRE.")
endif()

endif() # end of enable batch solvers