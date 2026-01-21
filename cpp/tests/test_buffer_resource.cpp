/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <span>
#include <sstream>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/owning_wrapper.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

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
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    auto ret = br.allocate(size, stream, reservation);
    if (size > 0) {
        ret->write_access([&](std::byte* ptr, rmm::cuda_stream_view s) {
            RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(ptr, 0, size, s));
        });
    }
    return ret;
}

TEST(BufferResource, ReservationOverbooking) {
    // Create a buffer resource that always have 10 KiB of available device memory.
    auto dev_mem_available = []() -> std::int64_t { return 10_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(),
        PinnedMemoryResource::Disabled,
        {{MemoryType::DEVICE, dev_mem_available}}
    };
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available memory.
    auto [reserve1, overbooking1] =
        br.reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Try to overbook.
    auto [reserve2, overbooking2] =
        br.reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve2.size(), 0);  // Reservation failed.
    EXPECT_EQ(overbooking2, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allow overbooking.
    auto [reserve3, overbooking3] =
        br.reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserve3.size(), 10_KiB);
    EXPECT_EQ(overbooking3, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // No host limit.
    auto [reserve4, overbooking4] =
        br.reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(overbooking4, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, 20_KiB), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 15_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // We are still overbooking.
    auto [reserve5, overbooking5] =
        br.reserve(MemoryType::DEVICE, 5_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserve5.size(), 5_KiB);
    EXPECT_EQ(overbooking5, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
}

TEST(BufferResource, ReservationReleasing) {
    // Create a buffer resource that always have 10 KiB of available host and device
    // memory.
    auto dev_mem_available = []() -> std::int64_t { return 10_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(),
        PinnedMemoryResource::Disabled,
        {{MemoryType::DEVICE, dev_mem_available}, {MemoryType::HOST, dev_mem_available}}
    };
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Reserve all available host and device memory.
    auto [reserve1, overbooking1] =
        br.reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    auto [reserve2, overbooking2] =
        br.reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(reserve2.size(), 10_KiB);
    EXPECT_EQ(overbooking2, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, 20_KiB), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // A reservation is released when it goes out of scope.
    {
        auto [reserve, overbooking] =
            br.reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::YES);
        EXPECT_EQ(reserve.size(), 10_KiB);
        EXPECT_EQ(overbooking, 10_KiB);
        EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
        EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 20_KiB);
    }
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
}

TEST(BufferResource, LimitAvailableMemory) {
    rmm::mr::cuda_memory_resource mr_cuda;
    RmmResourceAdaptor mr{mr_cuda};
    auto stream = cudf::get_default_stream();

    // Create a buffer resource that limit available device memory to 10 KiB.
    LimitAvailableMemory dev_mem_available{&mr, 10_KiB};
    BufferResource br{
        mr, PinnedMemoryResource::Disabled, {{MemoryType::DEVICE, dev_mem_available}}
    };
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available device memory.
    auto [reserve1, overbooking1] =
        br.reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::NO);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allocating a Buffer also requires a reservation, which are then released.
    auto dev_buf1 = zeros(br, 10_KiB, stream, reserve1);
    EXPECT_EQ(dev_buf1->mem_type(), MemoryType::DEVICE);
    EXPECT_EQ(dev_buf1->size, 10_KiB);
    EXPECT_EQ(reserve1.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    // Insufficent reservation for the allocation.
    EXPECT_THROW(zeros(br, 10_KiB, stream, reserve1), std::overflow_error);

    // Freeing a buffer increases the available but the reserved memory is unchanged.
    dev_buf1.reset();
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);

    // Moving buffers between memory types requires a reservation.
    auto [reserve2, overbooking2] =
        br.reserve(MemoryType::DEVICE, 10_KiB, AllowOverbooking::YES);
    auto dev_buf2 = zeros(br, 10_KiB, stream, reserve2);
    EXPECT_EQ(dev_buf2->mem_type(), MemoryType::DEVICE);
    auto [reserve3, overbooking3] =
        br.reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::YES);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    auto host_buf2 = br.move(std::move(dev_buf2), reserve3);
    EXPECT_EQ(host_buf2->mem_type(), MemoryType::HOST);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // Moving buffers to the same memory type accepts an empty reservation.
    auto host_buf3 = br.move(std::move(host_buf2), reserve3);
    EXPECT_EQ(host_buf3->mem_type(), MemoryType::HOST);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // The reservation must be of the correct memory type.
    auto [reserve4, overbooking4] =
        br.reserve(MemoryType::HOST, 10_KiB, AllowOverbooking::YES);
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
}

class BufferResourceReserveOrFailTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a buffer resource with limited device memory (10 KiB) and unlimited
        // host memory.
        cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        mr = std::make_unique<RmmResourceAdaptor>(*cuda_mr);
        br = std::make_unique<BufferResource>(
            *mr,
            PinnedMemoryResource::Disabled,
            std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{
                {MemoryType::DEVICE, LimitAvailableMemory{mr.get(), 10_KiB}}
            }
        );
    }

    std::unique_ptr<rmm::mr::cuda_memory_resource> cuda_mr;
    std::unique_ptr<RmmResourceAdaptor> mr;
    std::unique_ptr<BufferResource> br;
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
    EXPECT_EQ(br->memory_reserved(MemoryType::DEVICE), 5_KiB);
    EXPECT_THROW(
        std::ignore = br->reserve_or_fail(100_KiB, MemoryType::DEVICE), std::runtime_error
    );
}

TEST_F(BufferResourceReserveOrFailTest, HostType) {
    // Test reserve_or_fail with single host memory type
    auto res = br->reserve_or_fail(5_KiB, MemoryType::HOST);
    EXPECT_EQ(res.size(), 5_KiB);
    EXPECT_EQ(res.mem_type(), MemoryType::HOST);
    EXPECT_EQ(br->memory_reserved(MemoryType::HOST), 5_KiB);
}

TEST_F(BufferResourceReserveOrFailTest, MultipleTypes) {
    // just test the vector case. All other container types are tested in the static
    // assertions above.
    std::vector<MemoryType> types{MemoryType::DEVICE, MemoryType::HOST};
    auto res = br->reserve_or_fail(5_KiB, types);
    EXPECT_EQ(res.size(), 5_KiB);
    EXPECT_EQ(res.mem_type(), MemoryType::DEVICE);
    EXPECT_EQ(br->memory_reserved(MemoryType::DEVICE), 5_KiB);

    auto res1 = br->reserve_or_fail(10_KiB, types);  // this falls back to host
    EXPECT_EQ(res1.size(), 10_KiB);
    EXPECT_EQ(res1.mem_type(), MemoryType::HOST);
    EXPECT_EQ(br->memory_reserved(MemoryType::HOST), 10_KiB);
}

class BaseBufferResourceCopyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());
        stream = cudf::get_default_stream();

        // initialize the host pattern
        host_pattern.resize(buffer_size);
        for (std::size_t i = 0; i < host_pattern.size(); ++i) {
            host_pattern[i] = static_cast<uint8_t>(i % 256);
        }
    }

    std::unique_ptr<Buffer> create_and_initialize_buffer(
        MemoryType const mem_type, std::size_t const size
    ) {
        auto [alloc_reserve, alloc_overbooking] =
            br->reserve(mem_type, size, AllowOverbooking::NO);
        auto buf = br->allocate(size, stream, alloc_reserve);
        EXPECT_EQ(buf->mem_type(), mem_type);
        // copy the host pattern to the Buffer
        buf->write_access([&](std::byte* buf_data, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                buf_data, host_pattern.data(), size, cudaMemcpyDefault, stream
            ));
        });
        return buf;
    }

    static constexpr std::size_t buffer_size = 1024;  // 1 KiB

    std::unique_ptr<BufferResource> br;
    rmm::cuda_stream_view stream;

    std::vector<uint8_t> host_pattern;  // a predefined pattern for testing
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
        auto slice = br->allocate(stream, br->reserve_or_fail(length, dest_type));
        buffer_copy(
            *slice,
            *source,
            length,
            0,  // dst_offset
            std::ptrdiff_t(offset)  // src_offset
        );
        EXPECT_EQ(slice->mem_type(), dest_type);
        slice->stream().synchronize();
        EXPECT_TRUE(slice->is_latest_write_done());

        std::vector<uint8_t> verify_data(length);
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            verify_data.data(), slice->data(), length, cudaMemcpyDefault, stream
        ));
        stream.synchronize();
        verify_slice(verify_data, offset, length);
        return slice;
    }

    // verify the buffer is the same as the host pattern[offset:offset+length]
    void verify_slice(
        std::vector<uint8_t> const& data,
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
            *dest,
            *source,
            source->size,
            std::ptrdiff_t(dest_offset),  // dst_offset
            0  // src_offset
        );
        dest->stream().synchronize();
        EXPECT_TRUE(dest->is_latest_write_done());

        std::vector<uint8_t> verify_data_buf(length);
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            verify_data_buf.data(),
            dest->data() + dest_offset,
            length,
            cudaMemcpyDefault,
            stream
        ));
        stream.synchronize();
        verify_slice(verify_data_buf, 0, length);
    }

    // verify the slice of the buffer[offset:offset+length] is the same as the host
    // pattern
    void verify_slice(
        std::vector<uint8_t> const& data,
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
    auto dest = br->allocate(buffer_size, stream, dest_reserve);
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
        stream = cudf::get_default_stream();

        // Host pattern for initialization and verification
        host_pattern.resize(buffer_size);
        for (std::size_t i = 0; i < host_pattern.size(); ++i) {
            host_pattern[i] = static_cast<uint8_t>(i % 256);
        }

        // Setup br1 with statistics for its device memory
        mr_cuda1 = std::make_unique<rmm::mr::cuda_memory_resource>();
        mr1 = std::make_unique<RmmResourceAdaptor>(*mr_cuda1);
        br1 = std::make_unique<BufferResource>(mr1.get());

        // Setup br2 with statistics for its device memory
        mr_cuda2 = std::make_unique<rmm::mr::cuda_memory_resource>();
        mr2 = std::make_unique<RmmResourceAdaptor>(*mr_cuda2);
        br2 = std::make_unique<BufferResource>(mr2.get());
    }

    std::unique_ptr<Buffer> create_source_buffer() {
        auto [reserv1, ob1] =
            br1->reserve(MemoryType::DEVICE, buffer_size, AllowOverbooking::NO);
        auto buf1 = br1->allocate(buffer_size, stream, reserv1);
        EXPECT_EQ(reserv1.size(), 0);  // reservation should be consumed
        EXPECT_EQ(buf1->size, buffer_size);
        EXPECT_EQ(buf1->mem_type(), MemoryType::DEVICE);

        buf1->write_access([&](std::byte* buf1_data, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                buf1_data,
                host_pattern.data(),
                buffer_size,
                cudaMemcpyHostToDevice,
                stream
            ));
        });
        buf1->stream().synchronize();
        EXPECT_EQ(mr1->get_main_record().total(), buffer_size);
        return buf1;
    }

    void verify_memory_allocation(
        std::size_t expected_br1_total, std::size_t expected_br2_total
    ) {
        EXPECT_EQ(mr1->get_main_record().total(), expected_br1_total);
        EXPECT_EQ(mr2->get_main_record().total(), expected_br2_total);
    }

    std::size_t buffer_size;
    rmm::cuda_stream_view stream;
    std::vector<uint8_t> host_pattern;

    std::unique_ptr<rmm::mr::cuda_memory_resource> mr_cuda1;
    std::unique_ptr<rmm::mr::cuda_memory_resource> mr_cuda2;
    std::unique_ptr<RmmResourceAdaptor> mr1;
    std::unique_ptr<RmmResourceAdaptor> mr2;
    std::unique_ptr<BufferResource> br1;
    std::unique_ptr<BufferResource> br2;
};

TEST_F(BufferResourceDifferentResourcesTest, CopySlice) {
    constexpr std::ptrdiff_t slice_offset = 128;
    constexpr std::size_t slice_length = 512;

    auto buf1 = create_source_buffer();

    // Reserve memory for the slice on br2
    auto res2 = br2->reserve_or_fail(slice_length, MEMORY_TYPES);

    // Create slice of buf1 on br2
    auto buf2 = br2->allocate(slice_length, stream, res2);
    buffer_copy(
        *buf2,
        *buf1,
        slice_length,
        0,  // dst_offset
        slice_offset  // src_offset

    );
    EXPECT_EQ(buf2->size, slice_length);
    EXPECT_EQ(res2.size(), 0);  // reservation should be consumed
    buf2->stream().synchronize();

    // Verify memory allocation
    verify_memory_allocation(buffer_size, slice_length);
}

TEST_F(BufferResourceDifferentResourcesTest, Copy) {
    auto buf1 = create_source_buffer();

    // Create copy of buf1 on br2
    auto buf2 = br2->allocate(stream, br2->reserve_or_fail(buffer_size, MEMORY_TYPES));
    buffer_copy(*buf2, *buf1, buffer_size);
    EXPECT_EQ(buf2->size, buffer_size);
    buf2->stream().synchronize();

    // Verify memory allocation
    verify_memory_allocation(buffer_size, buffer_size);
}

class BufferCopyEdgeCases : public BaseBufferResourceCopyTest {};

TEST_F(BufferCopyEdgeCases, IllegalArguments) {
    constexpr std::size_t N = 1024;

    auto src = create_and_initialize_buffer(MemoryType::HOST, N);
    auto dst = br->allocate(stream, br->reserve_or_fail(N, MemoryType::HOST));

    // Negative offsets
    EXPECT_THROW(buffer_copy(*dst, *src, 10, -1, 0), std::invalid_argument);
    EXPECT_THROW(buffer_copy(*dst, *src, 10, 0, -1), std::invalid_argument);

    // Offsets beyond size
    EXPECT_THROW(
        buffer_copy(*dst, *src, 10, static_cast<std::ptrdiff_t>(N + 1), 0),
        std::invalid_argument
    );
    EXPECT_THROW(
        buffer_copy(*dst, *src, 10, 0, static_cast<std::ptrdiff_t>(N + 1)),
        std::invalid_argument
    );

    // Ranges out of bounds
    EXPECT_THROW(
        buffer_copy(*dst, *src, 16, static_cast<std::ptrdiff_t>(N - 8), 0),
        std::invalid_argument
    );
    EXPECT_THROW(
        buffer_copy(*dst, *src, 16, 0, static_cast<std::ptrdiff_t>(N - 8)),
        std::invalid_argument
    );
}

TEST_F(BufferCopyEdgeCases, ZeroSizeIsNoOp) {
    constexpr std::size_t N = 128;

    auto src = create_and_initialize_buffer(MemoryType::HOST, N);
    auto dst = br->allocate(stream, br->reserve_or_fail(N, MemoryType::HOST));

    // Pre-fill dst with a sentinel pattern
    std::vector<uint8_t> sent(N, 0xCD);
    dst->write_access([&](std::byte* dst_data, rmm::cuda_stream_view stream) {
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(dst_data, sent.data(), N, cudaMemcpyDefault, stream)
        );
    });
    EXPECT_NO_THROW(buffer_copy(*dst, *src, 0, 0, 0));
    dst->stream().synchronize();

    // dst unchanged
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(static_cast<uint8_t>(dst->data()[i]), 0xCD);
    }
}

TEST_F(BufferCopyEdgeCases, SameBufferIsDisallowed) {
    // Matches current implementation which rejects &dst == &src.
    constexpr std::size_t N = 64;

    auto buf = br->allocate(stream, br->reserve_or_fail(N, MemoryType::HOST));

    EXPECT_THROW(buffer_copy(*buf, *buf, 16, 0, 0), std::invalid_argument);
}
