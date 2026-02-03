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

    void verify_packed_data(cudf::table const& expect, auto& packed_data) {
        EXPECT_NE(packed_data, nullptr);
        EXPECT_NE(packed_data->metadata, nullptr);
        EXPECT_NE(packed_data->data, nullptr);

        if (num_rows_ == 0) {  // Skip unpacking for empty tables
            EXPECT_FALSE(packed_data->metadata->empty());
            return;
        }

        EXPECT_EQ(packed_data->data->mem_type(), mem_type_);
        EXPECT_FALSE(packed_data->empty());

        // if the destination memory type is host, we need to move the data to device
        if (!is_device_accessible(mem_type_)) {
            auto device_reservation =
                br_->reserve_or_fail(packed_data->data->size, MemoryType::DEVICE);
            packed_data->data =
                br_->move(std::move(packed_data->data), device_reservation);
        }

        auto unpacked = cudf::unpack(
            packed_data->metadata->data(),
            reinterpret_cast<std::uint8_t const*>(packed_data->data->data())
        );

        stream_.synchronize();

        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, unpacked);
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
        ::testing::Values(0, 100, 10000)  // empty, small, large
    ),
    [](const ::testing::TestParamInfo<std::tuple<MemoryType, std::size_t>>& info) {
        auto mem_type = std::get<0>(info.param);
        auto num_rows = std::get<1>(info.param);
        return std::string(to_string(mem_type)) + "_" + std::to_string(num_rows)
               + "_rows";
    }
);

TEST_P(CudfPackTest, PackAndUnpack) {
    cudf::table expect = random_table_with_index(seed, num_rows_, 0, 1000);

    auto cudf_packed = cudf::pack(expect, stream_, br_->device_mr());
    std::size_t packed_size = cudf_packed.gpu_data->size();

    auto reservation = br_->reserve_or_fail(packed_size, mem_type_);
    auto packed_data = pack(expect.view(), stream_, reservation);

    EXPECT_NO_FATAL_FAILURE(verify_packed_data(expect, packed_data));
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

    EXPECT_NO_FATAL_FAILURE(verify_packed_data(expect, packed_data));
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

    EXPECT_NO_FATAL_FAILURE(verify_packed_data(expect, packed_data));
}
