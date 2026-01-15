/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <gtest/gtest.h>

#include <cuda/memory>

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/integrations/cudf/utils.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
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

    // Device -> Device (moves data)
    auto on_gpu = unspill_partitions(std::move(input), br.get(), true);
    ASSERT_EQ(on_gpu.size(), 1);
    EXPECT_EQ(on_gpu[0].data->mem_type(), rapidsmpf::MemoryType::DEVICE);
    EXPECT_EQ(*on_gpu[0].metadata, metadata);

    // Device -> Host
    auto back_on_host = spill_partitions(std::move(on_gpu), br.get());
    ASSERT_EQ(back_on_host.size(), 1);
    EXPECT_EQ(back_on_host[0].data->mem_type(), rapidsmpf::MemoryType::HOST);
    EXPECT_EQ(*back_on_host[0].metadata, metadata);

    // Check that contents match original
    auto res = br->reserve_or_fail(back_on_host[0].data->size, MemoryType::HOST);
    auto actual = br->move_to_host_buffer(std::move(back_on_host[0].data), res);
    EXPECT_EQ(actual->copy_to_uint8_vector(), payload);
}

class NumOfRows_MemType : public ::testing::TestWithParam<std::tuple<int, MemoryType>> {
  protected:
    // cudf::chunked_pack requires at least a 1 MiB bounce buffer
    static constexpr size_t chunk_size = 1 << 20;
    static constexpr int64_t seed = 42;

    void setup_br(
        MemoryType mem_type,
        std::unordered_map<MemoryType, BufferResource::MemoryAvailable>&& memory_available
    ) {
        if (rapidsmpf::is_pinned_memory_resources_supported()) {
            pinned_mr = std::make_shared<rapidsmpf::PinnedMemoryResource>();
        } else {
            pinned_mr = PinnedMemoryResource::Disabled;
        }

        if (mem_type == MemoryType::PINNED_HOST
            && pinned_mr == PinnedMemoryResource::Disabled)
        {
            GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
        }

        br = std::make_unique<BufferResource>(
            cudf::get_current_device_resource_ref(),
            pinned_mr,
            std::move(memory_available)
        );
    }

    void SetUp() override {
        std::tie(num_rows, mem_type) = GetParam();

        std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available;
        // disable all memory types except mem_type
        for (auto mt : MEMORY_TYPES) {
            if (mt != mem_type) {
                memory_available[mt] = []() { return 0; };
            }
        }
        setup_br(mem_type, std::move(memory_available));
        stream = cudf::get_default_stream();
    }

    void validate_packed_table(
        cudf::table_view const& input_table, PackedData&& packed_data
    ) {
        EXPECT_EQ(mem_type, packed_data.data->mem_type());

        auto to_device = std::make_unique<rmm::device_buffer>(
            packed_data.data->data(), packed_data.data->size, stream, br->device_mr()
        );
        stream.synchronize();

        cudf::packed_columns packed_columns(
            std::move(packed_data.metadata), std::move(to_device)
        );
        auto unpacked_table = cudf::unpack(packed_columns);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input_table, unpacked_table);
    }

    int num_rows;
    MemoryType mem_type;
    cudf::table input_table;
    std::unique_ptr<BufferResource> br;
    std::shared_ptr<rapidsmpf::PinnedMemoryResource> pinned_mr;
    rmm::cuda_stream_view stream;
};

class ChunkedPackTest : public NumOfRows_MemType {};

// test different `num_rows` and `MemoryType`.
INSTANTIATE_TEST_SUITE_P(
    ChunkedPack,
    ChunkedPackTest,
    ::testing::Combine(
        ::testing::Values(0, 9, 1'000, 1'000'000, 10'000'000),  // num rows
        ::testing::ValuesIn(MEMORY_TYPES)  // output memory type
    ),
    [](const testing::TestParamInfo<NumOfRows_MemType::ParamType>& info) {
        return "nrows_" + std::to_string(std::get<0>(info.param)) + "_type_"
               + MEMORY_TYPE_NAMES[static_cast<std::size_t>(std::get<1>(info.param))];
    }
);

TEST_P(ChunkedPackTest, chunked_pack) {
    cudf::table input_table = random_table_with_index(seed, num_rows, 0, 10);

    auto [bounce_buf_res, _] = br->reserve(MemoryType::DEVICE, chunk_size, true);
    auto bounce_buf = br->allocate(chunk_size, stream, bounce_buf_res);

    auto data_res =
        br->reserve_or_fail(estimated_memory_usage(input_table, stream), mem_type);

    auto chunked_packed = rapidsmpf::chunked_pack(input_table, *bounce_buf, data_res);

    EXPECT_NO_THROW(validate_packed_table(input_table, std::move(chunked_packed)));
}

class PackToHostTest : public NumOfRows_MemType {};

INSTANTIATE_TEST_SUITE_P(
    PackTableToHost,
    PackToHostTest,
    ::testing::Combine(
        ::testing::Values(0, 9, 1'000, 1'000'000, 10'000'000),  // num rows
        ::testing::Values(MemoryType::HOST)  // output memory type
    ),
    [](const testing::TestParamInfo<NumOfRows_MemType::ParamType>& info) {
        return "nrows_" + std::to_string(std::get<0>(info.param)) + "_type_"
               + MEMORY_TYPE_NAMES[static_cast<std::size_t>(std::get<1>(info.param))];
    }
);

// device table to host packed data using 1MB device buffer
TEST_P(PackToHostTest, pack_to_host_with_1MB_device_buffer) {
    // override br with just 1MB device memory.
    std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available{
        {rapidsmpf::MemoryType::DEVICE, [] { return 1 << 20; }}
    };
    setup_br(mem_type, std::move(memory_available));

    cudf::table input_table = random_table_with_index(seed, num_rows, 0, 10);

    auto data_res =
        br->reserve_or_fail(estimated_memory_usage(input_table, stream), mem_type);

    std::array device_type{MemoryType::DEVICE};
    auto packed_data = rapidsmpf::pack(input_table, stream, data_res, device_type);

    EXPECT_NO_THROW(validate_packed_table(input_table, std::move(*packed_data)));
}

// device table to host packed data using 1MB device buffer
TEST_P(PackToHostTest, pack_to_host_with_unlimited_device_buffer) {
    // override br with just 1MB device memory.
    std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available{
        {rapidsmpf::MemoryType::DEVICE,
         [] { return std::numeric_limits<int64_t>::max(); }}
    };
    setup_br(mem_type, std::move(memory_available));

    cudf::table input_table = random_table_with_index(seed, num_rows, 0, 10);

    auto data_res =
        br->reserve_or_fail(estimated_memory_usage(input_table, stream), mem_type);

    std::cout << data_res.size() << " "
              << rapidsmpf::total_packing_wiggle_room(input_table) << std::endl;

    std::array device_type{MemoryType::DEVICE};
    auto packed_data = rapidsmpf::pack(input_table, stream, data_res, device_type);

    EXPECT_NO_THROW(validate_packed_table(input_table, std::move(*packed_data)));
}

// device table to host packed data using 1MB pinned buffer
TEST_P(PackToHostTest, pack_to_host_with_1MB_pinned_buffer) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory resources are not supported on the system.";
        return;
    }

    std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available{
        {rapidsmpf::MemoryType::PINNED_HOST, [] { return 1 << 20; }}
    };
    setup_br(mem_type, std::move(memory_available));

    cudf::table input_table = random_table_with_index(seed, num_rows, 0, 10);

    auto data_res =
        br->reserve_or_fail(estimated_memory_usage(input_table, stream), mem_type);

    auto packed_data =
        rapidsmpf::pack(input_table, stream, data_res, DEVICE_ACCESSIBLE_MEMORY_TYPES);

    EXPECT_NO_THROW(validate_packed_table(input_table, std::move(*packed_data)));
}

// device table to host packed data using unlimited pinned memory
TEST_P(PackToHostTest, pack_to_host_with_unlimited_pinned_buffer) {
    if (!rapidsmpf::is_pinned_memory_resources_supported()) {
        GTEST_SKIP() << "Pinned memory resources are not supported on the system.";
        return;
    }

    std::unordered_map<MemoryType, BufferResource::MemoryAvailable> memory_available{
        {rapidsmpf::MemoryType::PINNED_HOST,
         [] { return std::numeric_limits<int64_t>::max(); }}
    };
    setup_br(mem_type, std::move(memory_available));

    cudf::table input_table = random_table_with_index(seed, num_rows, 0, 10);

    auto data_res =
        br->reserve_or_fail(estimated_memory_usage(input_table, stream), mem_type);

    auto packed_data =
        rapidsmpf::pack(input_table, stream, data_res, DEVICE_ACCESSIBLE_MEMORY_TYPES);

    EXPECT_NO_THROW(validate_packed_table(input_table, std::move(*packed_data)));
}
