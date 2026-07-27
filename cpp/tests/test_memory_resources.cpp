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

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
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

using rapidsmpf::experimental::AllowOverbooking;
using rapidsmpf::experimental::MemoryReservation;

class ReservationAwareResourceAdaptorTest : public ::testing::Test {
  protected:
    /// @brief Let the deallocations enqueued by the test complete.
    void TearDown() override {
        stream.synchronize();
    }

    static constexpr std::int64_t limit = 1 << 20;

    rmm::cuda_stream_view stream{rmm::cuda_stream_default};
    rapidsmpf::experimental::ReservationAwareResourceAdaptor adaptor{
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            rmm::mr::cuda_memory_resource{}
        },
        limit
    };
};

void synchronize_pool(rmm::cuda_stream_pool& pool) {
    for (size_t i = 0; i < pool.get_pool_size(); ++i) {
        pool.get_stream(i).synchronize();
    }
}

}  // namespace

TEST_F(ReservationAwareResourceAdaptorTest, ReserveMovesBytesFromAvailableToReserved) {
    EXPECT_EQ(adaptor.available(), limit);

    // Zero-sized reservations are allowed.
    EXPECT_NO_THROW(std::ignore = adaptor.reserve(0, AllowOverbooking::NO));

    auto res = adaptor.reserve(1024, AllowOverbooking::NO);
    EXPECT_EQ(adaptor, res.adaptor());  // points to the same adaptor
    EXPECT_EQ(res.overbooking(), 0);
    EXPECT_EQ(res.balance(), 1024);
    EXPECT_EQ(adaptor.total_reserved(), 1024);
    EXPECT_EQ(adaptor.available(), limit - 1024);
}

TEST_F(ReservationAwareResourceAdaptorTest, AllocatingKeepsAvailableUnchanged) {
    auto res = adaptor.reserve(1024, AllowOverbooking::NO);

    {
        rmm::device_buffer buf1{256, stream, res};
        // The bytes moved from `total_reserved` to `current_allocated`.
        EXPECT_EQ(res.balance(), 768);
        EXPECT_EQ(adaptor.total_reserved(), 768);
        EXPECT_EQ(adaptor.current_allocated(), 256);
        EXPECT_EQ(adaptor.available(), limit - 1024);

        rmm::device_buffer buf2{512, stream, res};
        // The bytes moved from `total_reserved` to `current_allocated`.
        EXPECT_EQ(res.balance(), 256);
        EXPECT_EQ(adaptor.total_reserved(), 256);
        EXPECT_EQ(adaptor.current_allocated(), 768);
        EXPECT_EQ(adaptor.available(), limit - 1024);
    }

    EXPECT_EQ(res.balance(), 1024);
    EXPECT_EQ(adaptor.current_allocated(), 0);
    EXPECT_EQ(adaptor.available(), limit - 1024);
}

TEST_F(ReservationAwareResourceAdaptorTest, AllocatingOnDifferentStreams) {
    rmm::cuda_stream_pool pool{2, rmm::cuda_stream::flags::non_blocking};
    auto res = adaptor.reserve(1024, AllowOverbooking::NO);

    {
        rmm::device_buffer buf1{256, pool.get_stream(), res};
        // The bytes moved from `total_reserved` to `current_allocated`.
        EXPECT_EQ(res.balance(), 768);
        EXPECT_EQ(adaptor.total_reserved(), 768);
        EXPECT_EQ(adaptor.current_allocated(), 256);
        EXPECT_EQ(adaptor.available(), limit - 1024);

        rmm::device_buffer buf2{512, pool.get_stream(), res};
        // The bytes moved from `total_reserved` to `current_allocated`.
        EXPECT_EQ(res.balance(), 256);
        EXPECT_EQ(adaptor.total_reserved(), 256);
        EXPECT_EQ(adaptor.current_allocated(), 768);
        EXPECT_EQ(adaptor.available(), limit - 1024);
    }

    EXPECT_EQ(res.balance(), 1024);
    EXPECT_EQ(adaptor.current_allocated(), 0);
    EXPECT_EQ(adaptor.available(), limit - 1024);

    synchronize_pool(pool);
}

TEST_F(ReservationAwareResourceAdaptorTest, ExceedingTheGrantThrows) {
    auto res = adaptor.reserve(1024, AllowOverbooking::NO);
    EXPECT_THROW((rmm::device_buffer{2048, stream, res}), rmm::out_of_memory);
    // The failed allocation left the reservation untouched.
    EXPECT_EQ(res.balance(), 1024);
    EXPECT_EQ(adaptor.current_allocated(), 0);
}

TEST_F(ReservationAwareResourceAdaptorTest, ZeroSizedReservationThrowsOnFirstByte) {
    auto res = adaptor.reserve(2 * limit, AllowOverbooking::NO);
    EXPECT_EQ(res.balance(), 0);
    EXPECT_EQ(res.overbooking(), limit);
    EXPECT_THROW((rmm::device_buffer{1, stream, res}), rmm::out_of_memory);
}

TEST_F(ReservationAwareResourceAdaptorTest, OverbookingIsGrantedWhenAllowed) {
    auto res = adaptor.reserve(2 * limit, AllowOverbooking::YES);
    EXPECT_EQ(res.balance(), 2 * limit);
    EXPECT_EQ(res.overbooking(), limit);
    EXPECT_EQ(adaptor.available(), -limit);
}

TEST_F(ReservationAwareResourceAdaptorTest, DestructionRefundsTheUnusedBalance) {
    {
        auto res = adaptor.reserve(1024, AllowOverbooking::NO);
        EXPECT_EQ(adaptor.total_reserved(), 1024);
    }
    EXPECT_EQ(adaptor.total_reserved(), 0);
    EXPECT_EQ(adaptor.available(), limit);
}

TEST_F(ReservationAwareResourceAdaptorTest, BufferOutlivesTheReservingScope) {
    {
        auto buf = [&] {
            auto res = adaptor.reserve(1024, AllowOverbooking::NO);
            return rmm::device_buffer{512, stream, res};
        }();

        // The buffer carries the reservation itself rather than the bare adaptor, so
        // the remaining balance is still reachable through it.
        auto mr = buf.memory_resource();
        auto* reservation = cuda::mr::resource_cast<MemoryReservation>(&mr);
        ASSERT_NE(reservation, nullptr);
        EXPECT_EQ(reservation->balance(), 512);

        // The buffer holds a copy of the reservation, so the unspent 512 bytes stay
        // reserved even though the reserving scope is gone.
        EXPECT_EQ(adaptor.current_allocated(), 512);
        EXPECT_EQ(adaptor.total_reserved(), 512);
        EXPECT_EQ(adaptor.available(), limit - 1024);

        // The buffer grows through the reservation, so it is still capped by it.
        EXPECT_THROW(buf.resize(2048, stream), rmm::out_of_memory);
    }

    // Destroying the buffer drops the last reference to the reservation.
    EXPECT_EQ(adaptor.current_allocated(), 0);
    EXPECT_EQ(adaptor.total_reserved(), 0);
    EXPECT_EQ(adaptor.available(), limit);
}
