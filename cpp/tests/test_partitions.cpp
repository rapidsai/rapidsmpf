/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

#include "utils.hpp"

using namespace rapidsmpf;

class NumOfPartitions : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {};

// test different `num_partitions` and `num_rows`.
INSTANTIATE_TEST_SUITE_P(
    Partitions,
    NumOfPartitions,
    testing::Combine(
        testing::Range(1, 10),  // num_partitions
        testing::Range(1, 100, 9)  // num_rows
    )
);

TEST_P(NumOfPartitions, partition_and_pack) {
    int const num_partitions = std::get<0>(GetParam());
    int const num_rows = std::get<1>(GetParam());
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();
    rapidsmpf::BufferResource br{mr()};

    cudf::table expect =
        random_table_with_index(seed, static_cast<std::size_t>(num_rows), 0, 10);

    auto chunks = rapidsmpf::partition_and_pack(
        expect, {1}, num_partitions, hash_fn, seed, stream, &br
    );

    // Convert to a vector
    std::vector<rapidsmpf::PackedData> chunks_vector;
    for (auto& [_, chunk] : chunks) {
        chunks_vector.push_back(std::move(chunk));
    }
    EXPECT_EQ(chunks_vector.size(), num_partitions);

    auto result = rapidsmpf::unpack_and_concat(std::move(chunks_vector), stream, &br);

    // Compare the input table with the result. We ignore the row order by
    // sorting by their index (first column).
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
}

TEST_P(NumOfPartitions, split_and_pack) {
    int const num_partitions = std::get<0>(GetParam());
    int const num_rows = std::get<1>(GetParam());
    std::int64_t const seed = 42;
    auto stream = cudf::get_default_stream();
    rapidsmpf::BufferResource br{cudf::get_current_device_resource_ref()};

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    std::vector<cudf::size_type> splits;
    for (int i = 1; i < num_partitions; ++i) {
        splits.emplace_back(i * num_rows / num_partitions);
    }

    auto chunks = rapidsmpf::split_and_pack(expect, splits, stream, &br);

    // Convert to a vector (restoring the original order).
    std::vector<rapidsmpf::PackedData> chunks_vector;
    for (int i = 0; i < num_partitions; ++i) {
        chunks_vector.emplace_back(std::move(chunks.at(i)));
    }
    EXPECT_EQ(chunks_vector.size(), num_partitions);

    auto result = rapidsmpf::unpack_and_concat(std::move(chunks_vector), stream, &br);

    // Compare the input table with the result.
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, *result);
}

class SpillingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        br = std::make_unique<BufferResource>(cudf::get_current_device_resource_ref());
        stream = cudf::get_default_stream();
    }

    std::unique_ptr<BufferResource> br;
    rmm::cuda_stream_view stream;
};

TEST_F(SpillingTest, SpillUnspillRoundtripPreservesDataAndMetadata) {
    std::vector<uint8_t> metadata{42, 99};
    std::vector<uint8_t> payload{10, 20, 30};

    // Create device input.
    std::vector<rapidsmpf::PackedData> input;
    input.push_back(create_packed_data(metadata, payload, stream, br.get()));

    // Device -> Host
    auto on_gpu = unspill_partitions(
        std::move(input), stream, br.get(), /*allow_overbooking=*/true
    );
    ASSERT_EQ(on_gpu.size(), 1);
    EXPECT_EQ(on_gpu[0].data->mem_type(), rapidsmpf::MemoryType::DEVICE);
    EXPECT_EQ(*on_gpu[0].metadata, metadata);

    // Host -> Device
    auto back_on_host = spill_partitions(std::move(on_gpu), stream, br.get());
    ASSERT_EQ(back_on_host.size(), 1);
    EXPECT_EQ(back_on_host[0].data->mem_type(), rapidsmpf::MemoryType::HOST);
    EXPECT_EQ(*back_on_host[0].metadata, metadata);

    // Check that contents match original
    auto actual = br->move_to_host_vector(std::move(back_on_host[0].data));
    EXPECT_EQ(*actual, payload);
}
