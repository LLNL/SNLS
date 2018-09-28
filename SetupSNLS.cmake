if (NOT SNLS_LOADED)
    set (SNLS_LOADED True)
    mark_as_advanced(SNLS_LOADED)

    set( SNLS_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "" FORCE )

    # if an explicit build dir was not specified, set a default.
    if( NOT SNLS_BUILD_DIR )
        set( SNLS_BUILD_DIR ${PROJECT_BINARY_DIR}/snls CACHE PATH "" FORCE )
    endif()

endif() # only load SNLS once!
