/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cuda/memory_resource>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>

namespace {
// Compile-time tests for the memory type preference helpers.
namespace memory_type_tests {
using namespace rapidsmpf;
using enum MemoryType;

// `std::ranges::equal` cannot deduce a braced list, so wrap it.
constexpr bool equal(
    std::span<MemoryType const> actual, std::initializer_list<MemoryType> expected
) {
    return std::ranges::equal(actual, expected);
}

// preference_between
static_assert(
    equal(preference_between(PINNED_HOST, HOST, MEMORY_TYPES), {PINNED_HOST, HOST})
);
static_assert(equal(preference_between(HOST, HOST, MEMORY_TYPES), {HOST}));
// `to` > `from` --> empty
static_assert(preference_between(HOST, DEVICE, MEMORY_TYPES).empty());
static_assert(preference_between(HOST, HOST, std::span<MemoryType>{}).empty());
// an unknown `from` or `to` should return an empty view, even if `mem_types` holds it
constexpr auto UNKNOWN = static_cast<MemoryType>(-1);
constexpr std::array with_unknown{DEVICE, UNKNOWN, HOST};
static_assert(preference_between(UNKNOWN, HOST, with_unknown).empty());
static_assert(preference_between(UNKNOWN, HOST, MEMORY_TYPES).empty());
static_assert(preference_between(HOST, UNKNOWN, MEMORY_TYPES).empty());
static_assert(preference_between(DEVICE, UNKNOWN, with_unknown).empty());

// preference_leq
static_assert(
    equal(preference_leq(DEVICE, MEMORY_TYPES), {DEVICE, PINNED_HOST, HOST, DISK})
);
static_assert(equal(preference_leq(HOST, MEMORY_TYPES), {HOST, DISK}));
static_assert(equal(preference_leq(DISK, MEMORY_TYPES), {DISK}));
static_assert(std::ranges::equal(
    preference_leq(DEVICE, ADDRESSABLE_MEMORY_TYPES), ADDRESSABLE_MEMORY_TYPES
));
static_assert(equal(preference_leq(HOST, ADDRESSABLE_MEMORY_TYPES), {HOST}));
static_assert(preference_leq(DISK, ADDRESSABLE_MEMORY_TYPES).empty());
// unknown memory type should return an empty view
static_assert(preference_leq(static_cast<MemoryType>(-1), MEMORY_TYPES).empty());

// preference_lt
static_assert(equal(preference_lt(DEVICE, MEMORY_TYPES), {PINNED_HOST, HOST, DISK}));
static_assert(equal(preference_lt(PINNED_HOST, MEMORY_TYPES), {HOST, DISK}));
static_assert(equal(preference_lt(HOST, MEMORY_TYPES), {DISK}));
static_assert(preference_lt(DISK, MEMORY_TYPES).empty());
static_assert(preference_lt(static_cast<MemoryType>(-1), MEMORY_TYPES).empty());
static_assert(
    equal(preference_lt(DEVICE, ADDRESSABLE_MEMORY_TYPES), {PINNED_HOST, HOST})
);
static_assert(equal(preference_lt(PINNED_HOST, ADDRESSABLE_MEMORY_TYPES), {HOST}));
static_assert(preference_lt(HOST, ADDRESSABLE_MEMORY_TYPES).empty());
static_assert(preference_lt(DISK, ADDRESSABLE_MEMORY_TYPES).empty());
}  // namespace memory_type_tests

std::vector<cuda::mr::any_resource<cuda::mr::host_accessible>> make_host_resources() {
    auto pinned_pool_properties = rapidsmpf::is_pinned_memory_resources_supported()
                                      ? rapidsmpf::PinnedPoolProperties{}
                                      : rapidsmpf::PinnedMemoryDisabled;
    auto br = rapidsmpf::BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), std::move(pinned_pool_properties)
    );
    std::vector<cuda::mr::any_resource<cuda::mr::host_accessible>> resources;
    resources.emplace_back(br->host_mr());
    if (auto pinned = br->try_pinned_mr(); pinned.has_value()) {
        resources.emplace_back(*pinned);
    }
    return resources;
}

std::vector<cuda::mr::any_resource<cuda::mr::device_accessible>> make_device_resources() {
    auto pinned_pool_properties = rapidsmpf::is_pinned_memory_resources_supported()
                                      ? rapidsmpf::PinnedPoolProperties{}
                                      : rapidsmpf::PinnedMemoryDisabled;
    auto br = rapidsmpf::BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), std::move(pinned_pool_properties)
    );
    std::vector<cuda::mr::any_resource<cuda::mr::device_accessible>> resources;
    resources.emplace_back(rmm::mr::cuda_memory_resource{});
    resources.emplace_back(rmm::mr::cuda_async_memory_resource{});
    if (auto pinned = br->try_pinned_mr(); pinned.has_value()) {
        resources.emplace_back(*pinned);
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
