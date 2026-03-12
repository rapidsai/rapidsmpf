/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

/**
 * @brief Create and set a RMM stack as the current device memory resource.
 *
 * @param name The name of the stack:
 *  - `cuda`: use the default CUDA memory resource.
 *  - `async`: use a CUDA async memory resource.
 *  - `pool`: use a memory pool backed by a CUDA memory resource.
 *  - `managed`: use a memory pool backed by a CUDA managed memory resource.
 * @return A owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<rmm::mr::device_memory_resource>
set_current_rmm_stack(std::string const& name) {
    std::shared_ptr<rmm::mr::device_memory_resource> ret;
    if (name == "cuda") {
        ret = std::make_shared<rmm::mr::cuda_memory_resource>();
    } else if (name == "async") {
        ret = std::make_shared<rmm::mr::cuda_async_memory_resource>();
    } else if (name == "managed") {
        ret = std::make_shared<rmm::mr::managed_memory_resource>();
    } else if (name == "pool") {
        ret = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
            std::make_shared<rmm::mr::cuda_memory_resource>(),
            rmm::percent_of_free_device_memory(80),
            rmm::percent_of_free_device_memory(80)
        );
    } else {
        RAPIDSMPF_FAIL("unknown RMM stack name: " + name);
    }
    // Note, RMM maintains two default resources, we set both here.
    rmm::mr::set_current_device_resource(ret.get());
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}

/**
 * @brief Create a statistics-enabled device memory resource with on the current RMM
 * stack.
 *
 * @return A owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<rapidsmpf::RmmResourceAdaptor>
set_device_mem_resource_with_stats() {
    auto ret = std::make_shared<rapidsmpf::RmmResourceAdaptor>(
        cudf::get_current_device_resource_ref()
    );
    rmm::mr::set_current_device_resource(ret.get());
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}
