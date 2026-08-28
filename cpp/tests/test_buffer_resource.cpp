/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <span>
#include <sstream>

#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

/**
 * @brief Allocate a Buffer and initialize its contents to zero.
 *
 * @param br Buffer resource used for allocation.
 * @param size Number of bytes to allocate.
 * @param stream CUDA stream associated with the allocation.
 * @param reservation Memory reservation used to track the allocation.
 * @return A unique pointer to the zero-initialized Buffer.
 */
std::unique_ptr<Buffer> zeros(
    BufferResource& br,
    std::size_t size,
    cuda::stream_ref stream,
    MemoryReservation& reservation
) {
    auto ret = br.make_buffer(size, stream, reservation);
    if (size > 0) {
        ret->write_access([&](std::byte* ptr, cuda::stream_ref s) {
            RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, 0, size, s.get()));
        });
    }
    return ret;
}

/**
 * @brief The outstanding reservations of a memory type, in bytes.
 *
 * @param br Buffer resource to query.
 * @param mem_type The target memory type.
 * @return The reserved bytes.
 */
std::int64_t reserved_bytes(BufferResource const& br, MemoryType mem_type) {
    return br.memory_available(mem_type) - br.memory_available_for_reservation(mem_type);
}

TEST(BufferResource, ReservationOverbooking) {
    // Create a buffer resource that always reports 10 KiB of available device memory.
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 10_KiB}}
    );
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // Book all available memory.
    auto [reserve1, overbooking1] =
        br->reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // Try to overbook.
    auto [reserve2, overbooking2] =
        br->reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve2.size(), 0);  // Reservation failed.
    EXPECT_EQ(overbooking2, 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // Allow overbooking.
    auto [reserve3, overbooking3] =
        br->reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserve3.size(), 10_KiB);
    EXPECT_EQ(overbooking3, 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // No host limit.
    auto [reserve4, overbooking4] =
        br->reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(overbooking4, 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br->release(reserve1, 20_KiB), rapidsmpf::reservation_error);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br->release(reserve1, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 15_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // We are still overbooking.
    auto [reserve5, overbooking5] =
        br->reserve(MemoryType::DEVICE, 5_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserve5.size(), 5_KiB);
    EXPECT_EQ(overbooking5, 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);
}

TEST(BufferResource, ReservationReleasing) {
    // Create a buffer resource that always reports 10 KiB of available host and device
    // memory.
    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        PinnedMemoryDisabled,
        {{MemoryType::DEVICE, 10_KiB}, {MemoryType::HOST, 10_KiB}}
    );
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // Reserve all available host and device memory.
    auto [reserve1, overbooking1] =
        br->reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    auto [reserve2, overbooking2] =
        br->reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(reserve2.size(), 10_KiB);
    EXPECT_EQ(overbooking2, 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br->release(reserve1, 20_KiB), rapidsmpf::reservation_error);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br->release(reserve1, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);

    // A reservation is released when it goes out of scope.
    {
        auto [reserve, overbooking] =
            br->reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::YES);
        EXPECT_EQ(reserve.size(), 10_KiB);
        EXPECT_EQ(overbooking, 10_KiB);
        EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);
        EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 20_KiB);
    }
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);
}

TEST(BufferResource, MemoryLimit) {
    rmm::mr::cuda_memory_resource mr_cuda;
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    // Create a buffer resource that limits available device memory to 10 KiB.
    auto br = BufferResource::create(
        mr_cuda, PinnedMemoryDisabled, {{MemoryType::DEVICE, 10_KiB}}
    );
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // Book all available device memory.
    auto [reserve1, overbooking1] =
        br->reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0);

    // Allocating a Buffer also requires a reservation, which are then released.
    auto dev_buf1 = zeros(*br, 10_KiB, stream, reserve1);
    EXPECT_EQ(dev_buf1->mem_type(), MemoryType::DEVICE);
    EXPECT_EQ(dev_buf1->size, 10_KiB);
    EXPECT_EQ(reserve1.size(), 0);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 0);

    // Insufficent reservation for the allocation.
    EXPECT_THROW(zeros(*br, 10_KiB, stream, reserve1), rapidsmpf::reservation_error);

    // Freeing a buffer increases the available but the reserved memory is unchanged.
    dev_buf1.reset();
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0_KiB);

    // Moving buffers between memory types requires a reservation.
    auto [reserve2, overbooking2] =
        br->reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::YES);
    auto dev_buf2 = zeros(*br, 10_KiB, stream, reserve2);
    EXPECT_EQ(dev_buf2->mem_type(), MemoryType::DEVICE);
    auto [reserve3, overbooking3] =
        br->reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 0);

    auto host_buf2 = br->move(std::move(dev_buf2), reserve3);
    EXPECT_EQ(host_buf2->mem_type(), MemoryType::HOST);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 10_KiB);

    // Moving buffers to the same memory type accepts an empty reservation.
    auto host_buf3 = br->move(std::move(host_buf2), reserve3);
    EXPECT_EQ(host_buf3->mem_type(), MemoryType::HOST);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 0_KiB);
    EXPECT_EQ(br->memory_available(MemoryType::DEVICE), 10_KiB);

    // The reservation must be of the correct memory type.
    auto [reserve4, overbooking4] =
        br->reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);
}

class PinnedMaxPoolSizeReservationLimitTest
    : public ::testing::TestWithParam<std::optional<std::size_t>> {};

TEST_P(PinnedMaxPoolSizeReservationLimitTest, TwoReservations) {
    if (!is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory not supported on this system";
    }

    auto const max_pool_size = GetParam();
    // if max_pool_size is not set or 0, the pool is unbounded.
    auto const expect_second_succeeds = [&] { return max_pool_size.value_or(0) == 0; };

    rmm::mr::cuda_memory_resource cuda_mr;

    // Wire the PINNED_HOST limit to the pool's max_pool_size (or unlimited if the
    // pool is unbounded) so reservations respect the same ceiling as allocations.
    std::unordered_map<MemoryType, std::int64_t> memory_limits;
    if (max_pool_size.has_value() && *max_pool_size > 0) {
        memory_limits[MemoryType::PINNED_HOST] = safe_cast<std::int64_t>(*max_pool_size);
    }
    auto br = BufferResource::create(
        cuda_mr,
        PinnedPoolProperties{.max_pool_size = max_pool_size},
        std::move(memory_limits)
    );
    ASSERT_TRUE(br->try_pinned_mr().has_value());

    // First 1 KiB reservation always succeeds.
    auto [r1, ob1] = br->reserve(MemoryType::PINNED_HOST, 1_KiB, AllowOverbooking::NO);
    EXPECT_EQ(r1.size(), 1_KiB);
    EXPECT_EQ(ob1, 0);

    // Second 1 KiB reservation succeeds only when the pool is unbounded.
    auto [r2, ob2] = br->reserve(MemoryType::PINNED_HOST, 1_KiB, AllowOverbooking::NO);
    EXPECT_EQ(r2.size(), expect_second_succeeds() ? 1_KiB : 0);
    EXPECT_EQ(ob2, expect_second_succeeds() ? 0 : 1_KiB);
}

INSTANTIATE_TEST_SUITE_P(
    PinnedMaxPoolSize,
    PinnedMaxPoolSizeReservationLimitTest,
    ::testing::Values(
        std::optional<std::size_t>{std::nullopt},
        std::optional<std::size_t>{0},
        std::optional<std::size_t>{1_KiB}
    )
);

TEST(BufferResource, AllocStatistics) {
    rmm::mr::cuda_memory_resource mr_cuda;
    auto stats = Statistics::create();
    bool const pinned_available = is_pinned_memory_resources_supported();
    auto br = BufferResource::create(
        mr_cuda,
        pinned_available ? PinnedPoolProperties{} : PinnedMemoryDisabled,
        {},
        std::nullopt,
        std::make_shared<StreamPool>(1),
        stats
    );
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    constexpr std::size_t device_size = 4_KiB;
    constexpr std::size_t pinned_size = 8_KiB;
    constexpr std::size_t host_size = 16_KiB;

    // Allocate device memory twice.
    {
        auto [r, _] = br->reserve(MemoryType::DEVICE, device_size, AllowOverbooking::YES);
        br->make_buffer(device_size, stream, r);
    }
    {
        auto [r, _] = br->reserve(MemoryType::DEVICE, device_size, AllowOverbooking::YES);
        br->make_buffer(device_size, stream, r);
    }
    // Allocate pinned_host memory once (if available).
    if (pinned_available) {
        auto [r, _] =
            br->reserve(MemoryType::PINNED_HOST, pinned_size, AllowOverbooking::YES);
        br->make_buffer(pinned_size, stream, r);
    }
    // Allocate host memory once.
    {
        auto [r, _] = br->reserve(MemoryType::HOST, host_size, AllowOverbooking::YES);
        br->make_buffer(host_size, stream, r);
    }

    stream.sync();

    // device: 2 allocations of device_size each.
    auto const dev_bytes = stats->get_stat("alloc-device-bytes");
    EXPECT_EQ(dev_bytes.count(), 2u);
    EXPECT_EQ(dev_bytes.value(), static_cast<double>(2 * device_size));

    // pinned_host: 1 allocation of pinned_size (if available).
    if (pinned_available) {
        auto const pinned_bytes = stats->get_stat("alloc-pinned_host-bytes");
        EXPECT_EQ(pinned_bytes.count(), 1u);
        EXPECT_EQ(pinned_bytes.value(), static_cast<double>(pinned_size));
        EXPECT_EQ(stats->get_stat("alloc-pinned_host-time").count(), 1u);
    }

    // host: 1 allocation of host_size.
    auto const host_bytes = stats->get_stat("alloc-host-bytes");
    EXPECT_EQ(host_bytes.count(), 1u);
    EXPECT_EQ(host_bytes.value(), static_cast<double>(host_size));

    // timing stats should have the same count as bytes stats.
    EXPECT_EQ(stats->get_stat("alloc-device-time").count(), 2u);
    EXPECT_EQ(stats->get_stat("alloc-host-time").count(), 1u);
}

class BufferResourceReserveOrFailTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a buffer resource with limited device memory (10 KiB) and unlimited
        // host memory. BufferResource auto-wraps mr_cuda for allocation tracking.
        br = BufferResource::create(
            mr_cuda,
            PinnedMemoryDisabled,
            std::unordered_map<MemoryType, std::int64_t>{{MemoryType::DEVICE, 10_KiB}}
        );
    }

    rmm::mr::cuda_memory_resource mr_cuda;
    std::shared_ptr<BufferResource> br;
};

// Static assertions to verify that various container types can be used with
// reserve_or_fail
static_assert(
    std::convertible_to<std::ranges::range_value_t<decltype(MEMORY_TYPES)>, MemoryType>
);
static_assert(
    std::convertible_to<std::ranges::range_value_t<std::vector<MemoryType>>, MemoryType>
);
static_assert(
    std::convertible_to<std::ranges::range_value_t<std::span<MemoryType>>, MemoryType>
);
static_assert(std::convertible_to<
              std::ranges::range_value_t<std::initializer_list<MemoryType>>,
              MemoryType>);

TEST_F(BufferResourceReserveOrFailTest, DeviceType) {
    // Test reserve_or_fail with single device memory type
    auto res = br->reserve_or_fail(5_KiB, MemoryType::DEVICE);
    EXPECT_EQ(res.size(), 5_KiB);
    EXPECT_EQ(res.mem_type(), MemoryType::DEVICE);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);
    EXPECT_THROW(
        std::ignore = br->reserve_or_fail(100_KiB, MemoryType::DEVICE), std::runtime_error
    );
}

TEST_F(BufferResourceReserveOrFailTest, Split) {
    auto res = br->reserve_or_fail(5_KiB, MemoryType::DEVICE);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);

    {
        auto sub = res.split(2_KiB);
        EXPECT_EQ(sub.size(), 2_KiB);
        EXPECT_EQ(sub.mem_type(), MemoryType::DEVICE);
        EXPECT_EQ(sub.br(), br.get());
        EXPECT_EQ(res.size(), 3_KiB);
        // Splitting only moves bytes between the two reservations.
        EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);

        EXPECT_THROW(std::ignore = res.split(4_KiB), rapidsmpf::reservation_error);
        EXPECT_EQ(res.size(), 3_KiB);  // The failed split changed nothing.
    }
    // The sub-reservation released its bytes when it went out of scope.
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 3_KiB);
}

TEST_F(BufferResourceReserveOrFailTest, HostType) {
    // Test reserve_or_fail with single host memory type
    auto res = br->reserve_or_fail(5_KiB, MemoryType::HOST);
    EXPECT_EQ(res.size(), 5_KiB);
    EXPECT_EQ(res.mem_type(), MemoryType::HOST);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 5_KiB);
}

TEST_F(BufferResourceReserveOrFailTest, MultipleTypes) {
    // just test the vector case. All other container types are tested in the static
    // assertions above.
    std::vector<MemoryType> types{MemoryType::DEVICE, MemoryType::HOST};
    auto res = br->reserve_or_fail(5_KiB, types);
    EXPECT_EQ(res.size(), 5_KiB);
    EXPECT_EQ(res.mem_type(), MemoryType::DEVICE);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::DEVICE), 5_KiB);

    auto res1 = br->reserve_or_fail(10_KiB, types);  // this falls back to host
    EXPECT_EQ(res1.size(), 10_KiB);
    EXPECT_EQ(res1.mem_type(), MemoryType::HOST);
    EXPECT_EQ(reserved_bytes(*br, MemoryType::HOST), 10_KiB);
}

class BaseBufferResourceCopyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
        stream = cuda::stream_ref{cudaStreamLegacy};

        // initialize the host pattern
        host_pattern.resize(buffer_size);
        for (std::size_t i = 0; i < host_pattern.size(); ++i) {
            host_pattern[i] = static_cast<std::uint8_t>(i % 256);
        }
    }

    std::unique_ptr<Buffer> create_and_initialize_buffer(
        MemoryType const mem_type, std::size_t const size
    ) {
        auto [alloc_reserve, alloc_overbooking] =
            br->reserve(mem_type, size, AllowOverbooking::NO);
        auto buf = br->make_buffer(size, stream, alloc_reserve);
        EXPECT_EQ(buf->mem_type(), mem_type);
        buf->write_access([&](std::byte* buf_data, cuda::stream_ref stream) {
            RAPIDSMPF_CUDA_TRY(
                cuda_memcpy_async(buf_data, host_pattern.data(), size, stream)
            );
        });
        buf->latest_write_event().host_wait();
        return buf;
    }

    static constexpr std::size_t buffer_size = 1024;  // 1 KiB

    std::shared_ptr<BufferResource> br;
    cuda::stream_ref stream{cudaStreamLegacy};

    std::vector<std::uint8_t> host_pattern;  // a predefined pattern for testing
};

struct CopySliceParams {
    std::size_t offset;
    std::size_t length;
};

// SliceCopyTestParams is a tuple of (source_type, dest_type, params)
using SliceCopyTestParams = std::tuple<MemoryType, MemoryType, CopySliceParams>;

class BufferResourceCopySliceTest
    : public BaseBufferResourceCopyTest,
      public ::testing::WithParamInterface<SliceCopyTestParams> {
  protected:
    std::unique_ptr<Buffer> copy_slice_and_verify(
        MemoryType const dest_type,
        std::unique_ptr<Buffer> const& source,
        std::size_t const offset,
        std::size_t const length
    ) {
        auto slice = br->make_buffer(stream, br->reserve_or_fail(length, dest_type));
        buffer_copy(
            br->statistics(),
            *slice,
            *source,
            length,
            0,  // dst_offset
            std::ptrdiff_t(offset)  // src_offset
        );
        EXPECT_EQ(slice->mem_type(), dest_type);
        slice->stream().sync();
        EXPECT_TRUE(slice->is_latest_write_done());

        std::vector<std::uint8_t> verify_data(length);
        RAPIDSMPF_CUDA_TRY(
            cuda_memcpy_async(verify_data.data(), slice->data(), length, stream)
        );
        stream.sync();
        verify_slice(verify_data, offset, length);
        return slice;
    }

    // verify the buffer is the same as the host pattern[offset:offset+length]
    void verify_slice(
        std::vector<std::uint8_t> const& data,
        std::size_t const offset,
        std::size_t const length
    ) {
        EXPECT_EQ(data.size(), length);
        for (std::size_t i = 0; i < length; ++i) {
            EXPECT_EQ(data[i], host_pattern[offset + i]);
        }
    }
};

TEST_P(BufferResourceCopySliceTest, CopySlice) {
    auto [source_type, dest_type, params] = GetParam();
    auto src_buf = create_and_initialize_buffer(source_type, buffer_size);
    copy_slice_and_verify(dest_type, src_buf, params.offset, params.length);
}

INSTANTIATE_TEST_SUITE_P(
    CopySliceTests,
    BufferResourceCopySliceTest,
    ::testing::Combine(
        ::testing::Values(MemoryType::HOST, MemoryType::DEVICE),  // source type
        ::testing::Values(MemoryType::HOST, MemoryType::DEVICE),  // dest type
        ::testing::Values(
            CopySliceParams{0, 0},  // Empty slice at start
            CopySliceParams{0, 1024},  // Full buffer
            CopySliceParams{1024, 0},  // Empty slice at end
            CopySliceParams{11, 37},  // Small slice in middle
            CopySliceParams{256, 512}  // Larger slice in middle
        )
    ),
    [](const ::testing::TestParamInfo<SliceCopyTestParams>& info) {
        std::stringstream ss;
        ss << (std::get<0>(info.param) == MemoryType::HOST ? "Host" : "Device") << "To"
           << (std::get<1>(info.param) == MemoryType::HOST ? "Host" : "Device") << "_"
           << "off_" << std::get<2>(info.param).offset << "_"
           << "len_" << std::get<2>(info.param).length;
        return ss.str();
    }
);

struct CopyToParams {
    std::size_t source_size;
    std::size_t dest_offset;
};

// CopyToTestParams is a tuple of (source_type, dest_type, params)
using CopyToTestParams = std::tuple<MemoryType, MemoryType, CopyToParams>;

class BufferResourceCopyToTest : public BaseBufferResourceCopyTest,
                                 public ::testing::WithParamInterface<CopyToTestParams> {
  protected:
    void copy_and_verify(
        std::unique_ptr<Buffer> const& source,
        std::unique_ptr<Buffer>& dest,
        std::size_t const dest_offset
    ) {
        auto length = source->size;
        buffer_copy(
            br->statistics(),
            *dest,
            *source,
            source->size,
            std::ptrdiff_t(dest_offset),  // dst_offset
            0  // src_offset
        );
        dest->stream().sync();
        EXPECT_TRUE(dest->is_latest_write_done());

        std::vector<std::uint8_t> verify_data_buf(length);
        RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(
            verify_data_buf.data(), dest->data() + dest_offset, length, stream
        ));
        stream.sync();
        verify_slice(verify_data_buf, 0, length);
    }

    // verify the slice of the buffer[offset:offset+length] is the same as the host
    // pattern
    void verify_slice(
        std::vector<std::uint8_t> const& data,
        std::size_t const offset,
        std::size_t const length
    ) {
        EXPECT_GE(data.size(), offset + length);
        for (std::size_t i = 0; i < length; ++i) {
            EXPECT_EQ(data[offset + i], host_pattern[i]);
        }
    }
};

TEST_P(BufferResourceCopyToTest, CopyTo) {
    auto [source_type, dest_type, params] = BufferResourceCopyToTest::GetParam();
    auto source = create_and_initialize_buffer(source_type, params.source_size);
    auto [dest_reserve, dest_overbooking] =
        br->reserve(dest_type, buffer_size, AllowOverbooking::NO);
    auto dest = br->make_buffer(buffer_size, stream, dest_reserve);
    EXPECT_EQ(dest->mem_type(), dest_type);

    copy_and_verify(source, dest, params.dest_offset);
}

INSTANTIATE_TEST_SUITE_P(
    CopyToTests,
    BufferResourceCopyToTest,
    ::testing::Combine(
        ::testing::Values(MemoryType::HOST, MemoryType::DEVICE),  // source type
        ::testing::Values(MemoryType::HOST, MemoryType::DEVICE),  // dest type
        ::testing::Values(
            // source_size, dest_offset (dest_size = 1024)
            CopyToParams{1024, 0},  // Same sized buffers
            CopyToParams{503, 0},  // Copy to beginning
            CopyToParams{503, 503},  // Copy to end
            CopyToParams{503, 257},  // Copy to middle
            CopyToParams{0, 0},  // Empty copy to beginning
            CopyToParams{0, 1024},  // Empty copy to end
            CopyToParams{0, 503}  // Empty copy to middle
        )
    ),
    [](const ::testing::TestParamInfo<CopyToTestParams>& info) {
        auto source_type = std::get<0>(info.param);
        auto dest_type = std::get<1>(info.param);
        auto params = std::get<2>(info.param);
        std::stringstream ss;
        ss << (source_type == MemoryType::HOST ? "Host" : "Device") << "To"
           << (dest_type == MemoryType::HOST ? "Host" : "Device") << "_"
           << "src_" << params.source_size << "_"
           << "dst_off_" << params.dest_offset;
        return ss.str();
    }
);

class BufferResourceDifferentResourcesTest : public ::testing::Test {
  protected:
    void SetUp() override {
        buffer_size = 1_KiB;
        stream = cuda::stream_ref{cudaStreamLegacy};

        // Host pattern for initialization and verification
        host_pattern.resize(buffer_size);
        for (std::size_t i = 0; i < host_pattern.size(); ++i) {
            host_pattern[i] = static_cast<std::uint8_t>(i % 256);
        }

        // `BufferResource` wraps the device resource in an internal
        // `RmmResourceAdaptor` for allocation tracking, so just pass a vanilla
        // resource; `device_total()` reads back the per-resource record.
        br1 = BufferResource::create(rmm::mr::cuda_memory_resource{});
        br2 = BufferResource::create(rmm::mr::cuda_memory_resource{});
    }

    /// @brief Cumulative device bytes allocated through @p br.
    static std::int64_t device_total(BufferResource& br) {
        return br.device_mr_adaptor().get_main_record().total();
    }

    std::unique_ptr<Buffer> create_source_buffer() {
        auto [reserv1, ob1] =
            br1->reserve(MemoryType::DEVICE, buffer_size, AllowOverbooking::NO);
        auto buf1 = br1->make_buffer(buffer_size, stream, reserv1);
        EXPECT_EQ(reserv1.size(), 0);  // reservation should be consumed
        EXPECT_EQ(buf1->size, buffer_size);
        EXPECT_EQ(buf1->mem_type(), MemoryType::DEVICE);

        buf1->write_access([&](std::byte* buf1_data, cuda::stream_ref stream) {
            RAPIDSMPF_CUDA_TRY(
                cuda_memcpy_async(buf1_data, host_pattern.data(), buffer_size, stream)
            );
        });
        buf1->stream().sync();
        EXPECT_EQ(device_total(*br1), static_cast<std::int64_t>(buffer_size));
        return buf1;
    }

    void verify_memory_allocation(
        std::size_t expected_br1_total, std::size_t expected_br2_total
    ) {
        EXPECT_EQ(device_total(*br1), static_cast<std::int64_t>(expected_br1_total));
        EXPECT_EQ(device_total(*br2), static_cast<std::int64_t>(expected_br2_total));
    }

    std::size_t buffer_size;
    cuda::stream_ref stream{cudaStreamLegacy};
    std::vector<std::uint8_t> host_pattern;

    std::shared_ptr<BufferResource> br1;
    std::shared_ptr<BufferResource> br2;
};

TEST_F(BufferResourceDifferentResourcesTest, CopySlice) {
    constexpr std::ptrdiff_t slice_offset = 128;
    constexpr std::size_t slice_length = 512;

    auto buf1 = create_source_buffer();

    // Reserve memory for the slice on br2
    auto res2 = br2->reserve_or_fail(slice_length, MEMORY_TYPES);

    // Create slice of buf1 on br2
    auto buf2 = br2->make_buffer(slice_length, stream, res2);
    buffer_copy(
        br2->statistics(),
        *buf2,
        *buf1,
        slice_length,
        0,  // dst_offset
        slice_offset  // src_offset

    );
    EXPECT_EQ(buf2->size, slice_length);
    EXPECT_EQ(res2.size(), 0);  // reservation should be consumed
    buf2->stream().sync();

    // Verify memory allocation
    verify_memory_allocation(buffer_size, slice_length);
}

TEST_F(BufferResourceDifferentResourcesTest, Copy) {
    auto buf1 = create_source_buffer();

    // Create copy of buf1 on br2
    auto buf2 = br2->make_buffer(stream, br2->reserve_or_fail(buffer_size, MEMORY_TYPES));
    buffer_copy(br2->statistics(), *buf2, *buf1, buffer_size);
    EXPECT_EQ(buf2->size, buffer_size);
    buf2->stream().sync();

    // Verify memory allocation
    verify_memory_allocation(buffer_size, buffer_size);
}

class BufferCopyEdgeCases : public BaseBufferResourceCopyTest {};

TEST_F(BufferCopyEdgeCases, IllegalArguments) {
    constexpr std::size_t N = 1024;

    auto src = create_and_initialize_buffer(MemoryType::HOST, N);
    auto dst = br->make_buffer(stream, br->reserve_or_fail(N, MemoryType::HOST));
    auto statistics = br->statistics();

    // Negative offsets
    EXPECT_THROW(buffer_copy(statistics, *dst, *src, 10, -1, 0), std::invalid_argument);
    EXPECT_THROW(buffer_copy(statistics, *dst, *src, 10, 0, -1), std::invalid_argument);

    // Offsets beyond size
    EXPECT_THROW(
        buffer_copy(statistics, *dst, *src, 10, static_cast<std::ptrdiff_t>(N + 1), 0),
        std::invalid_argument
    );
    EXPECT_THROW(
        buffer_copy(statistics, *dst, *src, 10, 0, static_cast<std::ptrdiff_t>(N + 1)),
        std::invalid_argument
    );

    // Ranges out of bounds
    EXPECT_THROW(
        buffer_copy(statistics, *dst, *src, 16, static_cast<std::ptrdiff_t>(N - 8), 0),
        std::invalid_argument
    );
    EXPECT_THROW(
        buffer_copy(statistics, *dst, *src, 16, 0, static_cast<std::ptrdiff_t>(N - 8)),
        std::invalid_argument
    );
}

TEST_F(BufferCopyEdgeCases, ZeroSizeIsNoOp) {
    constexpr std::size_t N = 128;

    auto src = create_and_initialize_buffer(MemoryType::HOST, N);
    auto dst = br->make_buffer(stream, br->reserve_or_fail(N, MemoryType::HOST));

    // Pre-fill dst with a sentinel pattern
    std::vector<std::uint8_t> sent(N, 0xCD);
    dst->write_access([&](std::byte* dst_data, cuda::stream_ref stream) {
        RAPIDSMPF_CUDA_TRY(cuda_memcpy_async(dst_data, sent.data(), N, stream));
    });
    EXPECT_NO_THROW(buffer_copy(br->statistics(), *dst, *src, 0, 0, 0));
    dst->stream().sync();

    // dst unchanged
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(static_cast<std::uint8_t>(dst->data()[i]), 0xCD);
    }
}

TEST_F(BufferCopyEdgeCases, SameBufferIsDisallowed) {
    // Matches current implementation which rejects &dst == &src.
    constexpr std::size_t N = 64;

    auto buf = br->make_buffer(stream, br->reserve_or_fail(N, MemoryType::HOST));

    EXPECT_THROW(
        buffer_copy(br->statistics(), *buf, *buf, 16, 0, 0), std::invalid_argument
    );
}

TEST(BufferResource, DeviceMrKeepsBufferResourceAlive) {
    constexpr std::size_t N = 1024;

    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    std::weak_ptr<BufferResource> weak_br = br;
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    // Construct a device_buffer using the BR memory resource. Internally,
    // `rmm::device_buffer` stores the resource as an owning `cuda::mr::any_resource`,
    // which deep-copies the underlying `RmmResourceAdaptor`. Its
    // `BackRefMixin<BufferResource>` base promotes the installed weak ref to a
    // `shared_ptr<BufferResource>` during the copy.
    auto buf = std::make_unique<rmm::device_buffer>(N, stream, br->device_mr());

    // Drop the original shared_ptr. The buffer's internally stored owning memory
    // resource should be keeping the BR alive.
    br.reset();
    EXPECT_FALSE(weak_br.expired()) << "BR freed while buffer still holds it";

    EXPECT_NO_THROW(buf.reset());

    // After the buffer is destroyed, no shared ownership should remain. If the
    // original adaptor held a strong back-reference, or another ownership cycle
    // existed, the BR would still be alive here.
    EXPECT_TRUE(weak_br.expired()) << "BR not destructed, refcount cycle?";
}

TEST(BufferResource, HostMrKeepsBufferResourceAlive) {
    constexpr std::size_t N = 1024;

    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    std::weak_ptr<BufferResource> weak_br = br;
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    // Allocate a HOST buffer. The underlying `HostBuffer` stores the host memory
    // resource as an owning `any_resource`, which copies the `HostMemoryResource`.
    // Its `BackRefMixin<BufferResource>` base promotes the installed weak ref to a
    // `shared_ptr<BufferResource>` during the copy.
    auto buf = br->make_buffer(stream, br->reserve_or_fail(N, MemoryType::HOST));

    // Drop the original shared_ptr. The buffer should keep the BR alive.
    br.reset();
    EXPECT_FALSE(weak_br.expired()) << "BR freed while host buffer still holds it";

    EXPECT_NO_THROW(buf.reset());

    // After the buffer is destroyed, no shared ownership should remain.
    EXPECT_TRUE(weak_br.expired()) << "BR not destructed, refcount cycle?";
}

TEST(BufferResource, PinnedMrKeepsBufferResourceAlive) {
    if (!is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory not supported on this system";
    }
    constexpr std::size_t N = 1024;

    auto br = BufferResource::create(
        rmm::mr::get_current_device_resource_ref(), PinnedPoolProperties{}
    );
    std::weak_ptr<BufferResource> weak_br = br;
    auto stream = cuda::stream_ref{cudaStreamLegacy};

    // Allocate a PINNED_HOST buffer. The underlying `HostBuffer` stores the pinned
    // memory resource as an owning `any_resource`, which copies the
    // `PinnedMemoryResource`. Its `BackRefMixin<BufferResource>` base promotes the
    // installed weak ref to a `shared_ptr<BufferResource>` during the copy.
    auto buf = br->make_buffer(stream, br->reserve_or_fail(N, MemoryType::PINNED_HOST));

    // Drop the original shared_ptr. The buffer should keep the BR alive.
    br.reset();
    EXPECT_FALSE(weak_br.expired()) << "BR freed while pinned buffer still holds it";

    EXPECT_NO_THROW(buf.reset());

    // After the buffer is destroyed, no shared ownership should remain.
    EXPECT_TRUE(weak_br.expired()) << "BR not destructed, refcount cycle?";
}

TEST(RmmResourceAdaptor, EqualityAcrossCopiesAndAccessPaths) {
    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    any_device_resource copy1{br->device_mr()};
    any_device_resource copy2 = copy1;

    RmmResourceAdaptor owned_copy = br->device_mr_adaptor();
    any_device_resource owned_any{br->device_mr_adaptor()};

    // RmmResourceAdaptor == RmmResourceAdaptor (custom operator==).
    EXPECT_EQ(owned_copy, br->device_mr_adaptor());

    // any_resource == any_resource.
    EXPECT_EQ(copy1, copy2);
    EXPECT_EQ(copy1, owned_any);

    // any_resource == resource_ref (`device_mr()`).
    EXPECT_EQ(copy1, br->device_mr());
    EXPECT_EQ(owned_any, br->device_mr());
}

// Guarantee that when stats enabled, br->device_mr() reference gets properly casted to an
// RmmResourceAdaptor and used by the memory recorder.
TEST(BufferResource, DeviceMrIsAddressableByMemoryRecorder) {
    constexpr std::size_t kAllocBytes = 1_MiB;

    auto br = BufferResource::create(rmm::mr::get_current_device_resource_ref());
    auto stats = Statistics::create();
    ASSERT_TRUE(stats->enabled());

    {
        auto rec = stats->create_memory_recorder(br->device_mr(), "br-scope");
        rmm::device_buffer buf{
            kAllocBytes, cuda::stream_ref{cudaStreamLegacy}, br->device_mr()
        };
    }

    auto const& records = stats->get_memory_records();
    ASSERT_EQ(records.count("br-scope"), 1u);
    auto const& rec = records.at("br-scope");
    EXPECT_EQ(rec.num_calls, 1u);
    EXPECT_EQ(rec.global_peak, static_cast<std::int64_t>(kAllocBytes));
    EXPECT_EQ(rec.scoped.peak(), static_cast<std::int64_t>(kAllocBytes));
    EXPECT_EQ(rec.scoped.total(), static_cast<std::int64_t>(kAllocBytes));
}
