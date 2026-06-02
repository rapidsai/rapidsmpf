# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================


if(RAPIDSMPF_HAVE_CUPTI)
  add_executable(example_cupti_monitor "example_cupti_monitor.cpp")
  set_target_properties(
    example_cupti_monitor
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/examples"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    example_cupti_monitor PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                                  "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
  )
  target_link_libraries(
    example_cupti_monitor PRIVATE rapidsmpf::rapidsmpf $<TARGET_NAME_IF_EXISTS:conda_env>
                                  maybe_asan
  )
  install(
    TARGETS example_cupti_monitor
    COMPONENT testing
    DESTINATION bin/examples/librapidsmpf
    EXCLUDE_FROM_ALL
  )
endif()
