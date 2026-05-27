/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include <cuda/memory_resource>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <rapidsmpf/error.hpp>

/**
 * @brief Create and set a RMM memory resource as the current device resource.
 *
 * @param name The name of the resource:
 *  - `cuda`: use the default CUDA memory resource.
 *  - `async`: use a CUDA async memory resource.
 *  - `pool`: use a memory pool backed by a CUDA memory resource.
 *  - `managed`: use a CUDA managed memory resource.
 */
inline void set_current_rmm_resource(std::string const& name) {
    if (name == "cuda") {
        rmm::mr::set_current_device_resource(rmm::mr::cuda_memory_resource{});
    } else if (name == "async") {
        rmm::mr::set_current_device_resource(rmm::mr::cuda_async_memory_resource{});
    } else if (name == "managed") {
        rmm::mr::set_current_device_resource(rmm::mr::managed_memory_resource{});
    } else if (name == "pool") {
        rmm::mr::set_current_device_resource(
            rmm::mr::pool_memory_resource{
                rmm::mr::cuda_memory_resource{},
                rmm::percent_of_free_device_memory(80),
                rmm::percent_of_free_device_memory(80)
            }
        );
    } else {
        RAPIDSMPF_FAIL("unknown RMM resource name: " + name);
    }
}

/**
 * @brief Return the current device resource as a CCCL `any_resource`.
 *
 * Compatibility shim for benchmarks that previously wrapped the current device
 * resource in `RmmResourceAdaptor` to gain statistics. Tracking is now part of
 * `BufferResource` itself, so callers pass the returned `any_resource` directly
 * to a `BufferResource` constructor.
 *
 * @return The current device resource as a type-erased CCCL resource.
 */
[[nodiscard]] inline cuda::mr::any_resource<cuda::mr::device_accessible>
set_device_mem_resource_with_stats() {
    return cuda::mr::any_resource<cuda::mr::device_accessible>{
        rmm::mr::get_current_device_resource_ref()
    };
}
