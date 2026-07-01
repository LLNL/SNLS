# This script ensures BLT is loaded with reasonable defaults for SNLS.

if(NOT BLT_LOADED)
  if(DEFINED BLT_SOURCE_DIR)
    # Use external BLT if user specifies BLT_SOURCE_DIR
    if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "BLT_SOURCE_DIR=${BLT_SOURCE_DIR} does not contain SetupBLT.cmake")
    endif()
  else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt")

    if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "BLT submodule is not initialized. Run `git submodule update --init` in git repository or set BLT_SOURCE_DIR to external BLT.")
    endif()
  endif()

  # Set and check language standard
  set(BLT_CXX_STD "c++17" CACHE STRING "")

  if(("${BLT_CXX_STD}" STREQUAL "c++98") OR
     ("${BLT_CXX_STD}" STREQUAL "c++11") OR
     ("${BLT_CXX_STD}" STREQUAL "c++14"))
    message(FATAL_ERROR "SNLS requires a minimum C++ standard of c++17. Please set BLT_CXX_STD accordingly.")
  endif()

  # Build tests by default
  option(ENABLE_TESTS "Enables tests" ON)

  # Disable unused BLT features
  option(ENABLE_DOCS "Enables documentation" OFF)
  option(ENABLE_EXAMPLES "Enables examples" OFF)
  option(ENABLE_GIT "Enables Git support" OFF)
  option(ENABLE_DOXYGEN "Enables Doxygen support" OFF)
  option(ENABLE_SPHINX "Enables Sphinx support" OFF)
  option(ENABLE_CLANGAPPLYREPLACEMENTS "Enables clang-apply-replacements support" OFF)
  option(ENABLE_CLANGQUERY "Enables Clang-query support" OFF)
  option(ENABLE_CLANGTIDY "Enables clang-tidy support" OFF)
  option(ENABLE_CPPCHECK "Enables Cppcheck support" OFF)
  option(ENABLE_VALGRIND "Enables Valgrind support" OFF)
  option(ENABLE_ASTYLE "Enables AStyle support" OFF)
  option(ENABLE_CLANGFORMAT "Enables ClangFormat support" OFF)
  option(ENABLE_UNCRUSTIFY "Enables Uncrustify support" OFF)
  option(ENABLE_YAPF "Enables Yapf support" OFF)
  option(ENABLE_CMAKEFORMAT "Enables CMakeFormat support" OFF)
  option(ENABLE_FORTRAN "Enables Fortran language support" OFF)
  option(ENABLE_FRUIT "Enables Fortran unit testing framework" OFF)

  # Use newer approach for exporting BLT targets
  option(BLT_EXPORT_THIRDPARTY "Export BLT targets" OFF)

  # Load BLT
  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

  # Use newer approach for exporting BLT targets
  if (${BLT_VERSION} VERSION_GREATER_EQUAL 0.6.0)
    blt_install_tpl_setups(DESTINATION share/snls/cmake/)
  endif()
endif()
