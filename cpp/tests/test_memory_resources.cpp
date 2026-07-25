/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cuda/memory_resource>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/resource_types.hpp>
#include <rapidsmpf/reservation_aware_resource_adaptor.hpp>

namespace {

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

namespace {

constexpr std::int64_t limit = 1 << 20;

rapidsmpf::ReservationAwareResourceAdaptor make_adaptor() {
    return rapidsmpf::ReservationAwareResourceAdaptor{
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            rmm::mr::cuda_memory_resource{}
        },
        limit
    };
}

}  // namespace

TEST(ReservationAwareResourceAdaptor, ReserveMovesBytesFromAvailableToReserved) {
    auto adaptor = make_adaptor();
    EXPECT_EQ(adaptor.available(), limit);

    auto [res, overbooking] = adaptor.reserve(1024, false);
    EXPECT_EQ(overbooking, 0);
    EXPECT_EQ(res.size(), 1024);
    EXPECT_EQ(adaptor.total_reserved(), 1024);
    EXPECT_EQ(adaptor.available(), limit - 1024);
}

TEST(ReservationAwareResourceAdaptor, AllocatingKeepsAvailableUnchanged) {
    auto adaptor = make_adaptor();
    auto [res, _] = adaptor.reserve(1024, false);

    void* ptr = res.mr().allocate_sync(1024);
    EXPECT_NE(ptr, nullptr);
    // The bytes moved from `total_reserved` to `current_allocated`.
    EXPECT_EQ(res.size(), 0);
    EXPECT_EQ(adaptor.total_reserved(), 0);
    EXPECT_EQ(adaptor.current_allocated(), 1024);
    EXPECT_EQ(adaptor.available(), limit - 1024);

    res.mr().deallocate_sync(ptr, 1024);
    EXPECT_EQ(res.size(), 1024);
    EXPECT_EQ(adaptor.current_allocated(), 0);
    EXPECT_EQ(adaptor.available(), limit - 1024);
}

TEST(ReservationAwareResourceAdaptor, ExceedingTheGrantThrows) {
    auto adaptor = make_adaptor();
    auto [res, _] = adaptor.reserve(1024, false);
    EXPECT_THROW(std::ignore = res.mr().allocate_sync(2048), rmm::out_of_memory);
    // The failed allocation left the reservation untouched.
    EXPECT_EQ(res.size(), 1024);
    EXPECT_EQ(adaptor.current_allocated(), 0);
}

TEST(ReservationAwareResourceAdaptor, ZeroSizedReservationThrowsOnFirstByte) {
    auto adaptor = make_adaptor();
    auto [res, overbooking] = adaptor.reserve(2 * limit, false);
    EXPECT_EQ(res.size(), 0);
    EXPECT_EQ(overbooking, limit);
    EXPECT_THROW(std::ignore = res.mr().allocate_sync(1), rmm::out_of_memory);
}

TEST(ReservationAwareResourceAdaptor, OverbookingIsGrantedWhenAllowed) {
    auto adaptor = make_adaptor();
    auto [res, overbooking] = adaptor.reserve(2 * limit, true);
    EXPECT_EQ(res.size(), 2 * limit);
    EXPECT_EQ(overbooking, limit);
    EXPECT_EQ(adaptor.available(), -limit);
}

TEST(ReservationAwareResourceAdaptor, ReleaseRefundsTheUnusedBalance) {
    auto adaptor = make_adaptor();
    {
        auto [res, _] = adaptor.reserve(1024, false);
        EXPECT_EQ(adaptor.total_reserved(), 1024);
    }
    EXPECT_EQ(adaptor.total_reserved(), 0);
    EXPECT_EQ(adaptor.available(), limit);
}

TEST(ReservationAwareResourceAdaptor, EscapedAllocationOutlivesItsOwner) {
    auto adaptor = make_adaptor();
    void* ptr = nullptr;
    auto handle = [&] {
        auto [res, _] = adaptor.reserve(1024, false);
        ptr = res.mr().allocate_sync(512);
        return res.mr();  // a copy, keeping the reservation state alive
    }();

    // The owner is gone: the unused 512 bytes went back, the allocated 512 did not.
    EXPECT_EQ(adaptor.total_reserved(), 0);
    EXPECT_EQ(adaptor.current_allocated(), 512);

    // Further allocations fall through to the adaptor: tracked, but unreserved.
    void* extra = handle.allocate_sync(256);
    EXPECT_EQ(adaptor.current_allocated(), 768);
    EXPECT_EQ(adaptor.total_reserved(), 0);

    handle.deallocate_sync(extra, 256);
    handle.deallocate_sync(ptr, 512);
    EXPECT_EQ(adaptor.current_allocated(), 0);
    EXPECT_EQ(adaptor.total_reserved(), 0);
}
