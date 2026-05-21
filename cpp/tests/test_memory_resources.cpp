/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>

#include <gtest/gtest.h>

#include <cuda/memory_resource>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>

namespace {

std::vector<cuda::mr::any_resource<cuda::mr::host_accessible>> make_host_resources() {
    std::vector<cuda::mr::any_resource<cuda::mr::host_accessible>> resources;
    resources.emplace_back(rapidsmpf::HostMemoryResource{});
    if (rapidsmpf::is_pinned_memory_resources_supported()) {
        resources.emplace_back(*rapidsmpf::PinnedMemoryResource::make_if_available());
    }
    return resources;
}

std::vector<cuda::mr::any_resource<cuda::mr::device_accessible>> make_device_resources() {
    std::vector<cuda::mr::any_resource<cuda::mr::device_accessible>> resources;
    resources.emplace_back(rmm::mr::cuda_memory_resource{});
    resources.emplace_back(rmm::mr::cuda_async_memory_resource{});
    if (rapidsmpf::is_pinned_memory_resources_supported()) {
        resources.emplace_back(*rapidsmpf::PinnedMemoryResource::make_if_available());
    }
    return resources;
}

}  // namespace

TEST(MemoryResourceAccessibility, IsHostAccessible) {
    auto resources = make_host_resources();
    for (auto& mr : resources) {
        cuda::mr::resource_ref<cuda::mr::host_accessible> ref{mr};
        EXPECT_TRUE(rapidsmpf::is_host_accessible(ref));
        // PinnedMemoryResource is host- and device-accessible; the rest are host-only.
        if (cuda::mr::resource_cast<rapidsmpf::PinnedMemoryResource>(&mr) == nullptr) {
            EXPECT_FALSE(rapidsmpf::is_device_accessible(ref));
        }
    }
}

TEST(MemoryResourceAccessibility, IsDeviceAccessible) {
    auto resources = make_device_resources();
    for (auto& mr : resources) {
        cuda::mr::resource_ref<cuda::mr::device_accessible> ref{mr};
        EXPECT_TRUE(rapidsmpf::is_device_accessible(ref));
        // PinnedMemoryResource is host- and device-accessible; the rest are device-only.
        if (cuda::mr::resource_cast<rapidsmpf::PinnedMemoryResource>(&mr) == nullptr) {
            EXPECT_FALSE(rapidsmpf::is_host_accessible(ref));
        }
    }
}
