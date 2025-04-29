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

if(USE_RAJA_ONLY OR USE_BATCH_SOLVERS)

################################
# camp
################################

if (CAMP_DIR)
   find_package(camp REQUIRED CONFIG PATHS ${CAMP_DIR})
elseif(USE_BATCH_SOLVERS)
   message(FATAL_ERROR "CAMP_DIR was not provided. It is needed to find CAMP.")
endif()

################################
# RAJA
################################

if (RAJA_DIR)
   if (CAMP_DIR AND TARGET camp)
      set(camp_DIR ${CAMP_DIR})
   elseif(USE_BATCH_SOLVERS)
     message(FATAL_ERROR "RAJA is enabled but camp target not found")
   endif()
   find_package(RAJA REQUIRED CONFIG PATHS ${RAJA_DIR})
else()
   message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
endif()

endif()

if(USE_BATCH_SOLVERS)

################################
# chai
################################

if (CHAI_DIR)
   if (UMPIRE_DIR AND RAJA_DIR AND FMT_DIR)
      set(umpire_DIR ${UMPIRE_DIR})
      set(raja_DIR ${RAJA_DIR})
      set(fmt_DIR ${FMT_DIR})
    else()
      message(FATAL_ERROR "CHAI is enabled but raja/umpire/fmt directories are not defined")
   endif()  
   find_package(chai REQUIRED CONFIG PATHS ${CHAI_DIR})
else()
   message(FATAL_ERROR "CHAI_DIR was not provided. It is needed to find CHAI.")
endif()

################################
# fmt
################################

if (FMT_DIR)
   find_package(fmt CONFIG PATHS ${FMT_DIR})
else()
   message(WARNING "FMT_DIR was not provided. This is a requirement for camp as of v2024.02.0. Ignore this warning if using older versions of the RAJA Portability Suite")
endif()

################################
# UMPIRE
################################

if (DEFINED UMPIRE_DIR)
   find_package(umpire REQUIRED CONFIG PATHS ${UMPIRE_DIR})
else()
   message(FATAL_ERROR "UMPIRE_DIR was not provided. It is needed to find UMPIRE.")
endif()

endif() # end of enable batch solvers
