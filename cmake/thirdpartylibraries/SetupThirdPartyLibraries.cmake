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

################################
# camp
################################

if (camp_DIR)
   find_package(camp REQUIRED CONFIG PATHS ${camp_DIR})
else()
    message(FATAL_ERROR "camp_DIR was not provided. It is needed to find CAMP.")
endif()


################################
# chai
################################

if (DEFINED chai_DIR)
   find_package(chai REQUIRED CONFIG PATHS ${chai_DIR})
else()
    message(FATAL_ERROR "chai_DIR was not provided. It is needed to find CHAI.")
  endif()

################################
# fmt
################################

if (fmt_DIR)
   find_package(fmt REQUIRED CONFIG PATHS ${fmt_DIR})
else()
    message(FATAL_ERROR "fmt_DIR was not provided. It is needed to find CAMP.")
endif()
  

################################
# RAJA
################################

if (DEFINED raja_DIR)
   find_package(RAJA REQUIRED CONFIG PATHS ${raja_DIR})
else()
    message(FATAL_ERROR "raja_DIR was not provided. It is needed to find RAJA.")
endif()


################################
# UMPIRE
################################

if (DEFINED umpire_DIR)
   find_package(umpire REQUIRED CONFIG PATHS ${umpire_DIR})
else()
    message(FATAL_ERROR "umpire_DIR was not provided. It is needed to find UMPIRE.")
endif()

endif() # end of enable batch solvers
