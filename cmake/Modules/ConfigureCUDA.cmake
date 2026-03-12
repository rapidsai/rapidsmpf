# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND RAPIDSMPF_CXX_FLAGS -Wall -Werror -Wextra -Wsign-conversion -Wno-unknown-pragmas
       -Wno-missing-field-initializers -Wno-error=deprecated-declarations
  )
endif()

# Function to disable -Wsign-conversion for specific targets
function(disable_sign_conversion_warning target)
  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${target} PRIVATE -Wno-sign-conversion)
  endif()
endfunction()

list(APPEND RAPIDSMPF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
if(CUDA_WARNINGS_AS_ERRORS)
  list(APPEND RAPIDSMPF_CUDA_FLAGS -Werror=all-warnings)
else()
  list(APPEND RAPIDSMPF_CUDA_FLAGS -Werror=cross-execution-space-call)
endif()
list(
  APPEND
  RAPIDSMPF_CUDA_FLAGS
  -Xcompiler=-Wall,-Werror,-Wextra,-Wsign-conversion,-Wno-unknown-pragmas,-Wno-missing-field-initializers,-Wno-error=deprecated-declarations
)
# This warning needs to be suppressed because some parts of cudf instantiate templated CCCL
# functions in contexts where the resulting instantiations would have internal linkage (e.g. in
# anonymous namespaces). In such contexts, the visibility attribute on the template is ignored, and
# the compiler issues a warning. This is not a problem and will be fixed in future versions of CCCL.
list(APPEND RAPIDSMPF_CUDA_FLAGS -diag-suppress=1407)

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND RAPIDSMPF_CXX_FLAGS -Wno-deprecated-declarations)
  list(APPEND RAPIDSMPF_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# make sure we produce smallest binary size
list(APPEND RAPIDSMPF_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  list(APPEND RAPIDSMPF_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUDF: Building with debugging flags")
  list(APPEND RAPIDSMPF_CUDA_FLAGS -Xcompiler=-rdynamic)
endif()
