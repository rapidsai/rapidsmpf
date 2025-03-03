/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <string>

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <rapidsmp/error.hpp>

/**
 * @brief Create and set a RMM stack as the current device memory resource.
 *
 * @param name The name of the stack:
 *  - `cuda`: use the default cuda memory resource.
 *  - `async`: use an cuda async memory resource.
 *  - `pool`: use an memory pool backed by a cuda memory resource.
 * @return A owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<rmm::mr::device_memory_resource>
set_current_rmm_stack(std::string const& name) {
    std::shared_ptr<rmm::mr::device_memory_resource> ret;
    if (name == "cuda") {
        ret = std::make_shared<rmm::mr::cuda_memory_resource>();
    } else if (name == "async") {
        ret = std::make_shared<rmm::mr::cuda_async_memory_resource>();
    } else if (name == "pool") {
        ret = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
            std::make_shared<rmm::mr::cuda_memory_resource>(),
            rmm::percent_of_free_device_memory(80),
            rmm::percent_of_free_device_memory(80)
        );
    } else {
        RAPIDSMP_FAIL("unknown RMM stack name: " + name);
    }
    // Note, RMM maintains two default resources, we set both here.
    rmm::mr::set_current_device_resource(ret.get());
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}

using stats_dev_mem_resource =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

/**
 * @brief Create a statistics-enabled device memory resource with on the current RMM
 * stack.
 *
 * @return A owning memory resource, which must be kept alive.
 */
[[nodiscard]] inline std::shared_ptr<stats_dev_mem_resource>
set_device_mem_resource_with_stats() {
    auto ret =
        std::make_shared<stats_dev_mem_resource>(cudf::get_current_device_resource_ref());
    rmm::mr::set_current_device_resource(ret.get());
    rmm::mr::set_current_device_resource_ref(*ret);
    return ret;
}
