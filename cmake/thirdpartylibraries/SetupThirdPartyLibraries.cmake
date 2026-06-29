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
  if(NOT TARGET camp)
    find_package(camp REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${CAMP_DIR})
  endif()

  if(NOT TARGET RAJA)
    find_package(raja REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${RAJA_DIR})
  endif()
endif()

if(USE_BATCH_SOLVERS)
  if(NOT TARGET fmt::fmt and NOT TARGET fmt::fmt-header-only)
    find_package(fmt REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${FMT_DIR})
  endif()

  if(NOT TARGET umpire)
    find_package(umpire REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${UMPIRE_DIR})
  endif()

  if(NOT TARGET chai)
    find_package(chai REQUIRED CONFIG NO_DEFAULT_PATH PATHS ${CHAI_DIR})
  endif()
endif()
