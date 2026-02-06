/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <gtest/gtest.h>

#include <cudf/contiguous_split.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/integrations/cudf/pack.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

namespace {

/**
 * @brief Verify that packed data can be unpacked and matches the expected table.
 *
 * @param expect The expected table to compare against.
 * @param packed_data The packed data to verify.
 * @param expected_mem_type The expected memory type of the packed data.
 * @param br The buffer resource to use for moving data if needed.
 * @param stream The CUDA stream to use.
 */
void verify_packed_data(
    cudf::table const& expect,
    std::unique_ptr<PackedData>& packed_data,
    MemoryType expected_mem_type,
    BufferResource* br,
    rmm::cuda_stream_view stream
) {
    EXPECT_NE(packed_data, nullptr);
    EXPECT_NE(packed_data->metadata, nullptr);
    EXPECT_NE(packed_data->data, nullptr);

    if (expect.num_rows() == 0) {  // Skip unpacking for empty tables
        EXPECT_FALSE(packed_data->metadata->empty());
        return;
    }

    EXPECT_EQ(packed_data->data->mem_type(), expected_mem_type);
    EXPECT_FALSE(packed_data->empty());

    // copy to device to unpack
    rmm::device_buffer copy_data_buffer(
        packed_data->data->data(), packed_data->data->size, stream, br->device_mr()
    );
    stream.synchronize();

    auto unpacked = cudf::unpack(
        packed_data->metadata->data(),
        reinterpret_cast<std::uint8_t const*>(copy_data_buffer.data())
    );

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, unpacked);
}

}  // namespace

class CudfPackTest
    : public cudf::test::BaseFixtureWithParam<std::tuple<MemoryType, std::size_t>> {
  protected:
    void SetUp() override {
        std::tie(mem_type_, num_rows_) = GetParam();

        stream_ = cudf::get_default_stream();

        if (mem_type_ == MemoryType::PINNED_HOST
            && !is_pinned_memory_resources_supported())
        {
            GTEST_SKIP() << "Pinned memory resources are not supported on this system";
        }

        br_ = std::make_unique<BufferResource>(
            cudf::get_current_device_resource_ref(),
            PinnedMemoryResource::make_if_available()
        );
    }

    static constexpr std::int64_t seed = 42;
    MemoryType mem_type_;
    std::size_t num_rows_;
    rmm::cuda_stream_view stream_;
    std::unique_ptr<BufferResource> br_;
};

INSTANTIATE_TEST_SUITE_P(
    MemoryTypesAndRows,
    CudfPackTest,
    ::testing::Combine(
        ::testing::Values(MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST),
        ::testing::Values(0, 1, 100, 1000000)
    ),
    [](const ::testing::TestParamInfo<std::tuple<MemoryType, std::size_t>>& info) {
        auto mem_type = std::get<0>(info.param);
        auto num_rows = std::get<1>(info.param);
        return std::string(to_string(mem_type)) + "_" + std::to_string(num_rows);
    }
);

TEST_P(CudfPackTest, PackAndUnpack) {
    cudf::table expect = random_table_with_index(seed, num_rows_, 0, 1000);

    auto cudf_packed = cudf::pack(expect, stream_, br_->device_mr());
    std::size_t packed_size = cudf_packed.gpu_data->size();

    auto reservation = br_->reserve_or_fail(packed_size, mem_type_);
    auto packed_data = pack(expect.view(), stream_, reservation);

    EXPECT_NO_FATAL_FAILURE(
        verify_packed_data(expect, packed_data, mem_type_, br_.get(), stream_)
    );
}

// test chunked pack and unpack with device bounce buffer
TEST_P(CudfPackTest, ChunkedPackAndUnpackDevice) {
    cudf::table expect = random_table_with_index(seed, num_rows_, 0, 1000);

    auto cudf_packed = cudf::pack(expect, stream_, br_->device_mr());
    std::size_t packed_size = cudf_packed.gpu_data->size();

    auto reservation = br_->reserve_or_fail(packed_size, mem_type_);

    rmm::device_async_resource_ref pack_temp_mr = br_->device_mr();
    rmm::device_buffer bounce_buffer(1_MiB, stream_, pack_temp_mr);
    auto packed_data =
        chunked_pack(expect.view(), stream_, bounce_buffer, pack_temp_mr, reservation);

    EXPECT_NO_FATAL_FAILURE(
        verify_packed_data(expect, packed_data, mem_type_, br_.get(), stream_)
    );
}

// test chunked pack and unpack with pinned bounce buffer
TEST_P(CudfPackTest, ChunkedPackAndUnpackPinned) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Skipping test for non-pinned memory type";
    }

    cudf::table expect = random_table_with_index(seed, num_rows_, 0, 1000);

    auto cudf_packed = cudf::pack(expect, stream_, br_->device_mr());
    std::size_t packed_size = cudf_packed.gpu_data->size();

    auto reservation = br_->reserve_or_fail(packed_size, mem_type_);

    rmm::device_async_resource_ref pack_temp_mr = br_->pinned_mr_as_device();
    rmm::device_buffer bounce_buffer(1_MiB, stream_, pack_temp_mr);
    auto packed_data =
        chunked_pack(expect.view(), stream_, bounce_buffer, pack_temp_mr, reservation);

    EXPECT_NO_FATAL_FAILURE(
        verify_packed_data(expect, packed_data, mem_type_, br_.get(), stream_)
    );
}

/**
 * @brief Test pack<HOST> with zero device memory available.
 *
 * This test verifies that packing to host memory works when no device memory is
 * available for the bounce buffer, falling back to pinned memory. If pinned memory
 * is also unavailable, the test expects an exception.
 */
TEST(CudfPackHostTest, PackToHostWithZeroDeviceMemory) {
    static constexpr std::int64_t seed = 42;
    static constexpr std::size_t num_rows = 1000000;

    auto stream = cudf::get_default_stream();

    // Create a buffer resource with 0 device memory available.
    auto pinned_mr = PinnedMemoryResource::make_if_available();
    auto br = std::make_unique<BufferResource>(
        cudf::get_current_device_resource_ref(),
        pinned_mr,
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable>{
            {MemoryType::DEVICE, []() -> std::int64_t { return 0; }}
        }
    );

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 1000);

    // Get packed size using cudf::pack (we need device memory for this estimation).
    auto cudf_packed = cudf::pack(expect, stream, br->device_mr());
    std::size_t packed_size = cudf_packed.gpu_data->size();

    auto reservation = br->reserve_or_fail(packed_size, MemoryType::HOST);

    if (is_pinned_memory_resources_supported()) {
        auto packed_data = pack(expect.view(), stream, reservation);
        EXPECT_NO_FATAL_FAILURE(
            verify_packed_data(expect, packed_data, MemoryType::HOST, br.get(), stream)
        );
    } else {
        EXPECT_THROW(
            std::ignore = pack(expect.view(), stream, reservation), std::runtime_error
        );
    }
}
