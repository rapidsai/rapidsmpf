/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <gtest/gtest.h>

#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/integrations/cudf/pack.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

namespace {

/// @brief Bounce buffer size for all tests: 1 MiB.
constexpr std::size_t kBounceBufferSize = 1_MiB;

/// @brief Compute number of rows for a given table size (single int64 column).
[[nodiscard]] cudf::size_type rows_for_size(std::size_t table_size_bytes) {
    return static_cast<cudf::size_type>(table_size_bytes / sizeof(std::int64_t));
}

}  // namespace

class BaseTablePackerTest : public ::testing::Test {
  protected:
    void SetUp(MemoryType dest_mem_type) {
        stream_ = cudf::get_default_stream();
        if (rapidsmpf::is_pinned_memory_resources_supported()) {
            br_ = std::make_unique<BufferResource>(
                cudf::get_current_device_resource_ref(),
                std::make_shared<PinnedMemoryResource>()
            );
        } else if (dest_mem_type == MemoryType::PINNED_HOST) {
            GTEST_SKIP() << "Pinned memory resources are not supported on this system";
        } else {
            br_ =
                std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());
        }
        packer_ = std::make_unique<TablePacker>(kBounceBufferSize, stream_, *br_);
    }

    void SetUp() override {
        SetUp(MemoryType::DEVICE);
    }

    void TearDown() override {
        packer_.reset();
        br_.reset();
    }

    rmm::cuda_stream_view stream_;
    std::unique_ptr<BufferResource> br_;
    std::unique_ptr<TablePacker> packer_;
};

/// @brief Test parameters: destination memory type and table size.
using TablePackerParams = std::tuple<MemoryType, std::size_t>;

/**
 * @brief Parameterized test fixture for TablePacker tests.
 *
 * Parameters are table size in bytes and destination buffer memory type.
 */
class TablePackerTest : public BaseTablePackerTest,
                        public ::testing::WithParamInterface<TablePackerParams> {
  protected:
    void SetUp() override {
        BaseTablePackerTest::SetUp(std::get<0>(GetParam()));
    }
};

// Parameterize with memory types (DEVICE, PINNED_HOST, HOST) and table sizes (0, 1KB,
// 1MB, 10MB)
INSTANTIATE_TEST_SUITE_P(
    TableSizesAndMemTypes,
    TablePackerTest,
    ::testing::Combine(
        ::testing::Values(MemoryType::DEVICE, MemoryType::PINNED_HOST, MemoryType::HOST),
        ::testing::Values(0, 1_KiB, 1_MiB, 10_MiB)
    ),
    [](::testing::TestParamInfo<TablePackerParams> const& info) {
        return std::string(to_string(std::get<0>(info.param))) + "_"
               + std::to_string(std::get<1>(info.param));
    }
);

TEST_P(TablePackerTest, PackToDeviceAndUnpack) {
    auto [dest_mem_type, table_size] = GetParam();
    cudf::size_type const num_rows = rows_for_size(table_size);
    std::int64_t const seed = 42;

    cudf::table input_table = random_table_with_index(seed, num_rows, 0, 1000);

    auto pack_op = packer_->aquire(input_table.view(), stream_, br_->device_mr());

    std::size_t const packed_size = pack_op.get_packed_size();
    // if table is empty, packed size should be 0
    EXPECT_TRUE(table_size == 0 || packed_size > 0);

    auto reservation = br_->reserve_or_fail(packed_size, dest_mem_type);
    auto dest_buf = br_->allocate(stream_, std::move(reservation));

    auto metadata = pack_op.build_metadata();
    EXPECT_NE(metadata, nullptr);
    EXPECT_FALSE(metadata->empty());

    std::size_t const bytes_packed = pack_op.pack(*dest_buf);
    EXPECT_EQ(bytes_packed, packed_size);

    pack_op.clear();

    // Use cudf::unpack and check if the result matches the input table.
    stream_.synchronize();

    auto device_reservation = br_->reserve_or_fail(packed_size, MemoryType::DEVICE);
    auto packed_columns = cudf::packed_columns{
        std::move(metadata),
        br_->move_to_device_buffer(std::move(dest_buf), device_reservation)
    };
    cudf::table_view unpacked_view = cudf::unpack(packed_columns);

    CUDF_TEST_EXPECT_TABLES_EQUAL(input_table.view(), unpacked_view);
}
