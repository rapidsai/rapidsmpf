/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <sstream>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

using namespace rapidsmpf;

constexpr std::size_t operator"" _KiB(unsigned long long n) {
    return n * (1 << 10);
}

TEST(BufferResource, ReservationOverbooking) {
    // Create a buffer resource that always have 10 KiB of available device memory.
    auto dev_mem_available = []() -> std::int64_t { return 10_KiB; };
    BufferResource br{
        cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, dev_mem_available}}
    };
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Try to overbook.
    auto [reserve2, overbooking2] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    EXPECT_EQ(reserve2.size(), 0);  // Reservation failed.
    EXPECT_EQ(overbooking2, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allow overbooking.
    auto [reserve3, overbooking3] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    EXPECT_EQ(reserve3.size(), 10_KiB);
    EXPECT_EQ(overbooking3, 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // No host limit.
    auto [reserve4, overbooking4] = br.reserve(MemoryType::HOST, 10_KiB, false);
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(overbooking4, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_THROW(br.release(reserve1, MemoryType::HOST, 10_KiB), std::invalid_argument);
    EXPECT_THROW(br.release(reserve4, MemoryType::DEVICE, 10_KiB), std::invalid_argument);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, MemoryType::DEVICE, 20_KiB), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 20_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, MemoryType::DEVICE, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 15_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // We are still overbooking.
    auto [reserve5, overbooking5] = br.reserve(MemoryType::DEVICE, 5_KiB, true);
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
        {{MemoryType::DEVICE, dev_mem_available}, {MemoryType::HOST, dev_mem_available}}
    };
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Reserve all available host and device memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    auto [reserve2, overbooking2] = br.reserve(MemoryType::HOST, 10_KiB, false);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(reserve2.size(), 10_KiB);
    EXPECT_EQ(overbooking2, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release the wrong memory type.
    EXPECT_THROW(br.release(reserve1, MemoryType::HOST, 10_KiB), std::invalid_argument);
    EXPECT_THROW(br.release(reserve2, MemoryType::DEVICE, 10_KiB), std::invalid_argument);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Cannot release more than the size of the reservation.
    EXPECT_THROW(br.release(reserve1, MemoryType::DEVICE, 20_KiB), std::overflow_error);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // Partial releasing a reservation.
    EXPECT_EQ(br.release(reserve1, MemoryType::DEVICE, 5_KiB), 5_KiB);
    EXPECT_EQ(reserve1.size(), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 5_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // A reservation is released when it goes out of scope.
    {
        auto [reserve, overbooking] = br.reserve(MemoryType::HOST, 10_KiB, true);
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
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr{mr_cuda};
    auto stream = cudf::get_default_stream();

    // Create a buffer resource that uses `statistics_resource_adaptor` to limit
    // available device memory to 10 KiB.
    LimitAvailableMemory dev_mem_available{&mr, 10_KiB};
    BufferResource br{mr, {{MemoryType::DEVICE, dev_mem_available}}};
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Book all available device memory.
    auto [reserve1, overbooking1] = br.reserve(MemoryType::DEVICE, 10_KiB, false);
    EXPECT_EQ(reserve1.size(), 10_KiB);
    EXPECT_EQ(overbooking1, 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0);

    // Allocating a Buffer also requires a reservation, which are then released.
    auto dev_buf1 = br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve1);
    EXPECT_EQ(dev_buf1->size, 10_KiB);
    EXPECT_EQ(reserve1.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    // Insufficent reservation for the allocation.
    EXPECT_THROW(
        br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve1), std::overflow_error
    );

    // Freeing a buffer increases the available but the reserved memory is unchanged.
    dev_buf1.reset();
    EXPECT_EQ(dev_mem_available(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);

    // Moving buffers between memory types requires a reservation.
    auto [reserve2, overbooking2] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    auto dev_buf2 = br.allocate(MemoryType::DEVICE, 10_KiB, stream, reserve2);
    auto [reserve3, overbooking3] = br.reserve(MemoryType::HOST, 10_KiB, true);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 0);

    auto host_buf2 = br.move(MemoryType::HOST, std::move(dev_buf2), stream, reserve3);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // Moving buffers to the same memory type accepts an empty reservation.
    auto host_buf3 = br.move(MemoryType::HOST, std::move(host_buf2), stream, reserve3);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 0_KiB);
    EXPECT_EQ(dev_mem_available(), 10_KiB);

    // But copying buffers always requires a reservation.
    EXPECT_THROW(
        br.copy(MemoryType::HOST, host_buf3, stream, reserve3), std::overflow_error
    );

    // The reservation must be of the correct memory type.
    auto [reserve4, overbooking4] = br.reserve(MemoryType::HOST, 10_KiB, true);
    EXPECT_THROW(
        br.copy(MemoryType::DEVICE, host_buf3, stream, reserve4), std::invalid_argument
    );
    EXPECT_EQ(reserve4.size(), 10_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);

    // With the correct memory type, we can copy the buffer.
    auto [reserve5, overbooking5] = br.reserve(MemoryType::DEVICE, 10_KiB, true);
    auto dev_buf3 = br.copy(MemoryType::DEVICE, host_buf3, stream, reserve5);
    EXPECT_EQ(dev_buf3->size, 10_KiB);
    EXPECT_EQ(reserve5.size(), 0);
    EXPECT_EQ(br.memory_reserved(MemoryType::DEVICE), 0_KiB);
    EXPECT_EQ(br.memory_reserved(MemoryType::HOST), 10_KiB);
    EXPECT_EQ(dev_mem_available(), 0);
}

TEST(BufferResource, CUDAEventTracking) {
    constexpr std::size_t buffer_size = 1 * 1024 * 1024;  // 1 MiB

    rmm::mr::cuda_memory_resource mr_cuda;
    auto stream = cudf::get_default_stream();

    // Create a buffer resource with no memory limits
    BufferResource br{mr_cuda, {}};

    // Helper lambdas for data initialization and verification
    auto initialize_data = [](std::vector<uint8_t>& data) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(i % 256);
        }
    };

    auto verify_data = [](const std::vector<uint8_t>& data) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(data[i], static_cast<uint8_t>(i % 256));
        }
    };

    // Test host-to-host copy (should not create an event)
    {
        auto host_data = std::make_unique<std::vector<uint8_t>>(1024);
        initialize_data(*host_data);
        auto host_buf = br.move(std::move(host_data));
        auto [host_reserve, host_overbooking] = br.reserve(MemoryType::HOST, 1024, false);
        auto host_copy = br.copy(MemoryType::HOST, host_buf, stream, host_reserve);
        host_copy->wait_for_ready();  // should be no-op
        EXPECT_TRUE(host_copy->is_ready());  // No event created

        // Verify the data
        auto verify_data_buf = std::make_unique<std::vector<uint8_t>>(1024);
        std::memcpy(verify_data_buf->data(), host_copy->data(), 1024);
        verify_data(*verify_data_buf);
    }

    // Test device-to-device copy (should create an event)
    {
        auto [alloc_reserve, alloc_overbooking] =
            br.reserve(MemoryType::DEVICE, buffer_size, false);
        auto dev_buf =
            br.allocate(MemoryType::DEVICE, buffer_size, stream, alloc_reserve);

        // Initialize device data with a pattern
        auto host_pattern = std::make_unique<std::vector<uint8_t>>(buffer_size);
        initialize_data(*host_pattern);
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
            dev_buf->data(),
            host_pattern->data(),
            buffer_size,
            cudaMemcpyHostToDevice,
            stream
        ));

        auto [copy_reserve, copy_overbooking] =
            br.reserve(MemoryType::DEVICE, buffer_size, false);
        auto dev_copy = br.copy(MemoryType::DEVICE, dev_buf, stream, copy_reserve);

        // Wait for copy to complete
        dev_copy->wait_for_ready();
        EXPECT_TRUE(dev_copy->is_ready());

        // Verify the data
        auto verify_data_buf = std::make_unique<std::vector<uint8_t>>(buffer_size);
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpy(
            verify_data_buf->data(), dev_copy->data(), buffer_size, cudaMemcpyDeviceToHost
        ));
        verify_data(*verify_data_buf);
    }

    // Test host-to-device copy (should create an event)
    {
        auto host_data = std::make_unique<std::vector<uint8_t>>(buffer_size);
        initialize_data(*host_data);
        auto host_buf = br.move(std::move(host_data));
        auto [dev_reserve, dev_overbooking] =
            br.reserve(MemoryType::DEVICE, buffer_size, false);

        auto dev_copy = br.copy(MemoryType::DEVICE, host_buf, stream, dev_reserve);

        // Wait for copy to complete
        dev_copy->wait_for_ready();
        EXPECT_TRUE(dev_copy->is_ready());

        // Verify the data
        auto verify_data_buf = std::make_unique<std::vector<uint8_t>>(buffer_size);
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpy(
            verify_data_buf->data(), dev_copy->data(), buffer_size, cudaMemcpyDeviceToHost
        ));
        verify_data(*verify_data_buf);
    }

    // Test device-to-host copy (should create an event)
    {
        auto [alloc_reserve, alloc_overbooking] =
            br.reserve(MemoryType::DEVICE, buffer_size, false);
        auto dev_buf =
            br.allocate(MemoryType::DEVICE, buffer_size, stream, alloc_reserve);

        // Initialize device data with a pattern
        auto host_pattern = std::make_unique<std::vector<uint8_t>>(buffer_size);
        initialize_data(*host_pattern);
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaMemcpyAsync(
            dev_buf->data(),
            host_pattern->data(),
            buffer_size,
            cudaMemcpyHostToDevice,
            stream
        ));

        auto [host_reserve, host_overbooking] =
            br.reserve(MemoryType::HOST, buffer_size, false);
        auto host_copy = br.copy(MemoryType::HOST, dev_buf, stream, host_reserve);

        // Wait for copy to complete
        host_copy->wait_for_ready();
        EXPECT_TRUE(host_copy->is_ready());

        // Verify the data
        auto verify_data_buf = std::make_unique<std::vector<uint8_t>>(buffer_size);
        std::memcpy(verify_data_buf->data(), host_copy->data(), buffer_size);
        verify_data(*verify_data_buf);
    }
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
        auto [alloc_reserve, alloc_overbooking] = br->reserve(mem_type, size, false);
        auto buf = br->allocate(mem_type, size, stream, alloc_reserve);

        if (mem_type == MemoryType::DEVICE) {
            // copy the host pattern to the device buffer
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                const_cast<void*>(buf->data()),
                host_pattern.data(),
                size,
                cudaMemcpyHostToDevice,
                stream
            ));
            // add an event to guarantee async copy is complete
            buf->override_event(std::make_shared<Buffer::Event>(stream));
        } else {
            // copy the host pattern to the host buffer
            std::memcpy(const_cast<void*>(buf->data()), host_pattern.data(), size);
        }

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
        auto [slice_reserve, slice_overbooking] = br->reserve(dest_type, length, false);
        auto slice =
            br->copy_slice(dest_type, source, offset, length, stream, slice_reserve);

        EXPECT_EQ(slice->mem_type(), dest_type);
        stream.synchronize();
        EXPECT_TRUE(slice->is_ready());

        if (dest_type == MemoryType::HOST) {
            verify_slice(*const_cast<const Buffer&>(*slice).host(), offset, length);
        } else {
            std::vector<uint8_t> verify_data(length);
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                verify_data.data(), slice->data(), length, cudaMemcpyDeviceToHost, stream
            ));
            stream.synchronize();
            verify_slice(verify_data, offset, length);
        }

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
        auto bytes_written = source->copy_to(*dest, dest_offset, stream);
        stream.synchronize();
        EXPECT_TRUE(dest->is_ready());
        EXPECT_EQ(bytes_written, length);

        if (dest->mem_type() == MemoryType::HOST) {
            verify_slice(*const_cast<const Buffer&>(*dest).host(), dest_offset, length);
        } else {
            std::vector<uint8_t> verify_data_buf(length);
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                verify_data_buf.data(),
                static_cast<uint8_t*>(dest->data()) + dest_offset,
                length,
                cudaMemcpyDeviceToHost,
                stream
            ));
            stream.synchronize();
            verify_slice(verify_data_buf, 0, length);
        }
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
    auto [dest_reserve, dest_overbooking] = br->reserve(dest_type, buffer_size, false);
    auto dest = br->allocate(dest_type, buffer_size, stream, dest_reserve);

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

TEST_F(BufferResourceCopyToTest, OutOfBounds) {
    // create a source and destination buffer with size 128 bytes
    auto source = create_and_initialize_buffer(MemoryType::HOST, 128);
    auto dest = create_and_initialize_buffer(MemoryType::HOST, 128);

    // try to copy with an offset that would exceed the destination buffer size
    EXPECT_THROW(std::ignore = source->copy_to(*dest, 1, stream), std::invalid_argument);
}
