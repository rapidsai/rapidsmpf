/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/memory_resource>

namespace rapidsmpf {

/// @brief Owning type-erased device memory resource.
using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

/// @brief Owning type-erased host- and device-accessible memory resource.
using any_host_device_resource =
    cuda::mr::any_resource<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Check whether a type-erased memory resource is host-accessible.
 *
 * Queries the resource's `dynamic_accessibility_property` and returns true if
 * the reported accessibility is host-only or host-and-device.
 *
 * @tparam Properties The property pack of the resource reference.
 * @param mr The memory resource reference to query.
 * @return True if the resource is host-accessible, false otherwise.
 */
template <typename... Properties>
[[nodiscard]] bool is_host_accessible(
    cuda::mr::resource_ref<Properties...> const& mr
) noexcept {
    // Unqualified call so ADL finds the hidden-friend `get_property` declared
    // inside CCCL's type-erasure machinery for `resource_ref`. The qualified
    // `cuda::mr::get_property` overload is SFINAE-disabled for type-erased
    // wrappers via `__disable_default_dynamic_accessibility_property`.
    auto const accessibility =
        get_property(mr, cuda::mr::dynamic_accessibility_property{});
    return accessibility == cuda::mr::__memory_accessibility::__host
           || accessibility == cuda::mr::__memory_accessibility::__host_device;
}

/**
 * @brief Check whether a type-erased memory resource is device-accessible.
 *
 * Queries the resource's `dynamic_accessibility_property` and returns true if
 * the reported accessibility is device-only or host-and-device.
 *
 * @tparam Properties The property pack of the resource reference.
 * @param mr The memory resource reference to query.
 * @return True if the resource is device-accessible, false otherwise.
 */
template <typename... Properties>
[[nodiscard]] bool is_device_accessible(
    cuda::mr::resource_ref<Properties...> const& mr
) noexcept {
    // See `is_host_accessible` for why the call is unqualified (ADL).
    auto const accessibility =
        get_property(mr, cuda::mr::dynamic_accessibility_property{});
    return accessibility == cuda::mr::__memory_accessibility::__device
           || accessibility == cuda::mr::__memory_accessibility::__host_device;
}

}  // namespace rapidsmpf
