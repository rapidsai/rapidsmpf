/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>
#include <variant>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>

/**
 * @brief Owns the RMM resource stack as a type-erased variant.
 *
 * Resources are value types with shared ownership; no base class exists.
 * We use a variant to hold whichever concrete resource was selected.
 */
struct RmmStack {
    using Variant = std::variant<
        rmm::mr::cuda_memory_resource,
        rmm::mr::cuda_async_memory_resource,
        rmm::mr::managed_memory_resource,
        rmm::mr::pool_memory_resource>;

    Variant resource;
};

/**
 * @brief Create and set a RMM stack as the current device memory resource.
 *
 * @param name The name of the stack:
 *  - `cuda`: use the default CUDA memory resource.
 *  - `async`: use a CUDA async memory resource.
 *  - `pool`: use a memory pool backed by a CUDA memory resource.
 *  - `managed`: use a memory pool backed by a CUDA managed memory resource.
 * @return An owning RmmStack, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<RmmStack> set_current_rmm_stack(
    std::string const& name
) {
    auto stack = std::make_shared<RmmStack>();
    if (name == "cuda") {
        stack->resource.emplace<rmm::mr::cuda_memory_resource>();
    } else if (name == "async") {
        stack->resource.emplace<rmm::mr::cuda_async_memory_resource>();
    } else if (name == "managed") {
        stack->resource.emplace<rmm::mr::managed_memory_resource>();
    } else if (name == "pool") {
        rmm::mr::cuda_memory_resource cuda_mr;
        stack->resource.emplace<rmm::mr::pool_memory_resource>(
            cuda_mr,
            rmm::percent_of_free_device_memory(80),
            rmm::percent_of_free_device_memory(80)
        );
    } else {
        RAPIDSMPF_FAIL("unknown RMM stack name: " + name);
    }
    std::visit(
        [](auto& mr) { rmm::mr::set_current_device_resource_ref(mr); }, stack->resource
    );
    return stack;
}

/**
 * @brief Create a statistics-enabled device memory resource with on the current RMM
 * stack.
 *
 * @return An owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<rapidsmpf::RmmResourceAdaptor>
set_device_mem_resource_with_stats() {
    auto ret = std::make_shared<rapidsmpf::RmmResourceAdaptor>(
        cudf::get_current_device_resource_ref()
    );
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}
