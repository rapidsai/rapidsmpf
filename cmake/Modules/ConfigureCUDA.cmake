# =================================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# =================================================================================

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND RAPIDSMP_CXX_FLAGS -Wall -Werror -Wextra -Wsign-conversion -Wno-unknown-pragmas
       -Wno-missing-field-initializers -Wno-error=deprecated-declarations
  )
endif()

list(APPEND RAPIDSMP_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
if(CUDA_WARNINGS_AS_ERRORS)
  list(APPEND RAPIDSMP_CUDA_FLAGS -Werror=all-warnings)
else()
  list(APPEND RAPIDSMP_CUDA_FLAGS -Werror=cross-execution-space-call)
endif()
list(
  APPEND
  RAPIDSMP_CUDA_FLAGS
  -Xcompiler=-Wall,-Werror,-Wextra,-Wsign-conversion,-Wno-unknown-pragmas,-Wno-missing-field-initializes,-Wno-error=deprecated-declarations
)
# This warning needs to be suppressed because some parts of cudf instantiate templated CCCL
# functions in contexts where the resulting instantiations would have internal linkage (e.g. in
# anonymous namespaces). In such contexts, the visibility attribute on the template is ignored, and
# the compiler issues a warning. This is not a problem and will be fixed in future versions of CCCL.
list(APPEND RAPIDSMP_CUDA_FLAGS -diag-suppress=1407)

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND RAPIDSMP_CXX_FLAGS -Wno-deprecated-declarations)
  list(APPEND RAPIDSMP_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# make sure we produce smallest binary size
list(APPEND RAPIDSMP_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  list(APPEND RAPIDSMP_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUDF: Building with debugging flags")
  list(APPEND RAPIDSMP_CUDA_FLAGS -Xcompiler=-rdynamic)
endif()
