# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

#[=======================================================================[.rst:
get_pmix
--------

Find the PMIx (Process Management Interface - Exascale) library.

This module finds the PMIx library, which is typically provided by Slurm
or OpenPMIx installations. PMIx enables scalable process coordination
without requiring a shared filesystem.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``PMIx::PMIx``
  The PMIx library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``PMIx_FOUND``
  True if the system has the PMIx library.
``PMIx_VERSION``
  The version of the PMIx library which was found.
``PMIx_INCLUDE_DIRS``
  Include directories needed to use PMIx.
``PMIx_LIBRARIES``
  Libraries needed to link to PMIx.

Hints
^^^^^

The following variables can be set to help find PMIx:

``PMIx_ROOT``
  Root directory of PMIx installation.
``PMIX_ROOT``
  Alternative root directory variable.
``SLURM_ROOT``
  Slurm installation directory (PMIx may be bundled with Slurm).

#]=======================================================================]

# Extract PMIx version from header file. Sets PMIx_VERSION in parent scope if version can be
# determined.
function(_pmix_extract_version include_dir)
  if(NOT EXISTS "${include_dir}/pmix_version.h")
    return()
  endif()

  file(STRINGS "${include_dir}/pmix_version.h" _pmix_version_lines
       REGEX "#define[ \t]+PMIX_(MAJOR|MINOR|RELEASE)_VERSION"
  )

  foreach(_line ${_pmix_version_lines})
    if(_line MATCHES "#define[ \t]+PMIX_MAJOR_VERSION[ \t]+([0-9]+)")
      set(_pmix_major "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "#define[ \t]+PMIX_MINOR_VERSION[ \t]+([0-9]+)")
      set(_pmix_minor "${CMAKE_MATCH_1}")
    elseif(_line MATCHES "#define[ \t]+PMIX_RELEASE_VERSION[ \t]+([0-9]+)")
      set(_pmix_release "${CMAKE_MATCH_1}")
    endif()
  endforeach()

  if(DEFINED _pmix_major
     AND DEFINED _pmix_minor
     AND DEFINED _pmix_release
  )
    set(PMIx_VERSION
        "${_pmix_major}.${_pmix_minor}.${_pmix_release}"
        PARENT_SCOPE
    )
  elseif(DEFINED _pmix_major AND DEFINED _pmix_minor)
    set(PMIx_VERSION
        "${_pmix_major}.${_pmix_minor}"
        PARENT_SCOPE
    )
  endif()
endfunction()

# Create the PMIx::PMIx imported target and find optional dependencies.
function(_pmix_create_target library include_dir)
  if(TARGET PMIx::PMIx)
    return()
  endif()

  add_library(PMIx::PMIx UNKNOWN IMPORTED)
  set_target_properties(
    PMIx::PMIx PROPERTIES IMPORTED_LOCATION "${library}" INTERFACE_INCLUDE_DIRECTORIES
                                                         "${include_dir}"
  )

  # PMIx may have dependencies on libevent or hwloc. Try to find and link them if available.
  find_library(EVENT_CORE_LIBRARY event_core)
  find_library(EVENT_PTHREADS_LIBRARY event_pthreads)
  find_library(HWLOC_LIBRARY hwloc)

  set(_pmix_extra_libs "")
  foreach(_lib EVENT_CORE_LIBRARY EVENT_PTHREADS_LIBRARY HWLOC_LIBRARY)
    if(${_lib})
      list(APPEND _pmix_extra_libs "${${_lib}}")
    endif()
  endforeach()

  if(_pmix_extra_libs)
    set_property(
      TARGET PMIx::PMIx
      APPEND
      PROPERTY INTERFACE_LINK_LIBRARIES "${_pmix_extra_libs}"
    )
  endif()

  mark_as_advanced(
    PMIx_INCLUDE_DIR PMIx_LIBRARY EVENT_CORE_LIBRARY EVENT_PTHREADS_LIBRARY HWLOC_LIBRARY
  )
endfunction()

# Find and configure the PMIx library. Sets PMIx_FOUND, PMIx_VERSION, PMIx_INCLUDE_DIRS,
# PMIx_LIBRARIES in parent scope. Creates PMIx::PMIx imported target if found.
function(find_and_configure_pmix)
  # Return early if already found
  if(TARGET PMIx::PMIx)
    set(PMIx_FOUND
        TRUE
        PARENT_SCOPE
    )
    return()
  endif()

  # First try pkg-config (most reliable method)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_PMIx QUIET pmix)
  endif()

  # Find include directory
  find_path(
    PMIx_INCLUDE_DIR
    NAMES pmix.h
    HINTS ${PC_PMIx_INCLUDEDIR} ${PC_PMIx_INCLUDE_DIRS} ${PMIx_ROOT}/include $ENV{PMIx_ROOT}/include
          $ENV{PMIX_ROOT}/include ${SLURM_ROOT}/include $ENV{SLURM_ROOT}/include
    PATHS /usr/include /usr/local/include /opt/pmix/include /usr/include/slurm
          /usr/local/include/slurm
  )

  # Find library
  find_library(
    PMIx_LIBRARY
    NAMES pmix
    HINTS ${PC_PMIx_LIBDIR}
          ${PC_PMIx_LIBRARY_DIRS}
          ${PMIx_ROOT}/lib
          ${PMIx_ROOT}/lib64
          $ENV{PMIx_ROOT}/lib
          $ENV{PMIx_ROOT}/lib64
          $ENV{PMIX_ROOT}/lib
          $ENV{PMIX_ROOT}/lib64
          ${SLURM_ROOT}/lib
          ${SLURM_ROOT}/lib64
          $ENV{SLURM_ROOT}/lib
          $ENV{SLURM_ROOT}/lib64
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/pmix/lib /opt/pmix/lib64
  )

  # Get version from header if found
  if(PMIx_INCLUDE_DIR)
    _pmix_extract_version("${PMIx_INCLUDE_DIR}")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    PMIx
    REQUIRED_VARS PMIx_LIBRARY PMIx_INCLUDE_DIR
    VERSION_VAR PMIx_VERSION
  )

  if(PMIx_FOUND)
    _pmix_create_target("${PMIx_LIBRARY}" "${PMIx_INCLUDE_DIR}")
  endif()

  # Export results to parent scope
  set(PMIx_FOUND
      ${PMIx_FOUND}
      PARENT_SCOPE
  )
  if(DEFINED PMIx_VERSION)
    set(PMIx_VERSION
        ${PMIx_VERSION}
        PARENT_SCOPE
    )
  endif()
  set(PMIx_INCLUDE_DIRS
      ${PMIx_INCLUDE_DIR}
      PARENT_SCOPE
  )
  set(PMIx_LIBRARIES
      ${PMIx_LIBRARY}
      PARENT_SCOPE
  )
endfunction()
