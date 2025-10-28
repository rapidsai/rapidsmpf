/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/cudf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"


using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

using StreamingTableChunk = BaseStreamingFixture;

TEST_F(StreamingTableChunk, FromTable) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    TableChunk chunk{seq, std::make_unique<cudf::table>(expect), stream};
    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_TRUE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_EQ(chunk.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
}

TEST_F(StreamingTableChunk, TableChunkOwner) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    // Static because the deleter function is a void(*)(void*) which precludes the use of
    // a lambda with captures.
    static std::size_t num_deletions{0};
    auto deleter = [](void* p) {
        num_deletions++;
        delete static_cast<int*>(p);
    };
    auto make_chunk = [&](TableChunk::ExclusiveView exclusive_view) {
        return TableChunk{
            seq,
            expect,
            expect.alloc_size(),
            stream,
            OwningWrapper(new int, deleter),
            exclusive_view
        };
    };
    auto check_chunk = [&](TableChunk const& chunk, bool is_spillable) {
        EXPECT_EQ(chunk.sequence_number(), seq);
        EXPECT_EQ(chunk.stream().value(), stream.value());
        EXPECT_TRUE(chunk.is_available());
        EXPECT_EQ(chunk.is_spillable(), is_spillable);
        EXPECT_EQ(chunk.make_available_cost(), 0);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
    };
    {
        auto chunk = make_chunk(TableChunk::ExclusiveView::NO);
        check_chunk(chunk, false);
        EXPECT_EQ(num_deletions, 0);
    }
    EXPECT_EQ(num_deletions, 1);
    {
        auto msg = Message(
            std::make_unique<TableChunk>(make_chunk(TableChunk::ExclusiveView::NO))
        );
        EXPECT_EQ(num_deletions, 1);
    }
    EXPECT_EQ(num_deletions, 2);
    {
        auto msg = Message(
            std::make_unique<TableChunk>(make_chunk(TableChunk::ExclusiveView::YES))
        );
        auto chunk = msg.release<TableChunk>();
        check_chunk(chunk, true);
        EXPECT_EQ(num_deletions, 2);
    }
    EXPECT_EQ(num_deletions, 3);
    {
        auto chunk = make_chunk(TableChunk::ExclusiveView::YES);
        check_chunk(chunk, true);
        auto res = br->reserve_or_fail(
            chunk.data_alloc_size(MemoryType::DEVICE), MemoryType::DEVICE
        );
        // This is like spilling since the original `chunk` is ExclusiveView::YES and
        // overwritten.
        chunk = chunk.copy(br.get(), res);
        EXPECT_EQ(num_deletions, 4);
    }
}

TEST_F(StreamingTableChunk, FromPackedColumns) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed = cudf::pack(expect, stream);

    TableChunk chunk{
        seq, std::make_unique<cudf::packed_columns>(std::move(packed)), stream
    };

    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_TRUE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_EQ(chunk.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
}

TEST_F(StreamingTableChunk, FromPackedDataOnDevice) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed_columns = cudf::pack(expect, stream);

    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        br->move(std::move(packed_columns.gpu_data), stream)
    );
    TableChunk chunk{seq, std::move(packed_data)};

    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_FALSE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_THROW((void)chunk.table_view(), std::invalid_argument);

    // Eventhough the table isn't available, it is still all in device memory
    // so the memory cost of making it available is zero.
    EXPECT_EQ(chunk.make_available_cost(), 0);
}

TEST_F(StreamingTableChunk, FromPackedDataOnHost) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed_columns = cudf::pack(expect, stream);
    std::size_t const size = packed_columns.gpu_data->size();

    // Move the gpu_data to a Buffer (still device memory).
    auto gpu_data_on_device = br->move(std::move(packed_columns.gpu_data), stream);

    // Copy the GPU data to host memory.
    auto [res, _] = br->reserve(MemoryType::HOST, size, true);
    auto gpu_data_on_host = br->move(std::move(gpu_data_on_device), res);

    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata), std::move(gpu_data_on_host)
    );
    TableChunk chunk{seq, std::move(packed_data)};

    EXPECT_EQ(chunk.sequence_number(), seq);
    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_FALSE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_THROW((void)chunk.table_view(), std::invalid_argument);
    EXPECT_EQ(chunk.make_available_cost(), size);
}

TEST_F(StreamingTableChunk, DeviceToDeviceCopy) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    auto expect = random_table_with_index(seed, num_rows, 0, 10);

    rapidsmpf::streaming::TableChunk chunk{
        seq, std::make_unique<cudf::table>(expect), stream
    };
    EXPECT_TRUE(chunk.is_available());

    auto res = br->reserve_or_fail(
        chunk.data_alloc_size(MemoryType::DEVICE), MemoryType::DEVICE
    );
    auto chunk2 = chunk.copy(br.get(), res);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk2.table_view(), expect);
}

TEST_F(StreamingTableChunk, DeviceToHostRoundTripCopy) {
    constexpr unsigned int num_rows = 64;
    constexpr std::int64_t seed = 2025;
    constexpr std::uint64_t seq = 7;

    auto expect = random_table_with_index(seed, num_rows, 0, 5);

    TableChunk dev_chunk{seq, std::make_unique<cudf::table>(expect), stream};
    EXPECT_TRUE(dev_chunk.is_available());
    EXPECT_TRUE(dev_chunk.is_spillable());
    EXPECT_EQ(dev_chunk.sequence_number(), seq);
    EXPECT_EQ(dev_chunk.stream().value(), stream.value());
    EXPECT_EQ(dev_chunk.make_available_cost(), 0);

    // Copy to host memory -> new chunk should be unavailable.
    auto host_res = br->reserve_or_fail(
        dev_chunk.data_alloc_size(MemoryType::DEVICE), MemoryType::HOST
    );
    auto host_copy = dev_chunk.copy(br.get(), host_res);
    EXPECT_FALSE(host_copy.is_available());
    EXPECT_TRUE(host_copy.is_spillable());
    EXPECT_EQ(host_copy.sequence_number(), seq);
    EXPECT_EQ(host_copy.stream().value(), stream.value());
    EXPECT_GT(host_copy.make_available_cost(), 0);

    // Host to host copy.
    auto host_res2 = br->reserve_or_fail(
        host_copy.data_alloc_size(MemoryType::HOST), MemoryType::HOST
    );
    auto host_copy2 = host_copy.copy(br.get(), host_res2);
    EXPECT_FALSE(host_copy2.is_available());
    EXPECT_TRUE(host_copy2.is_spillable());
    EXPECT_EQ(host_copy2.sequence_number(), seq);
    EXPECT_EQ(host_copy2.stream().value(), stream.value());
    EXPECT_EQ(host_copy2.make_available_cost(), host_copy.make_available_cost());

    // Bring the new host copy back to device and verify equality.
    auto dev_res = br->reserve_or_fail(
        host_copy2.data_alloc_size(MemoryType::HOST), MemoryType::DEVICE
    );
    auto dev_back = host_copy2.make_available(dev_res);
    EXPECT_TRUE(dev_back.is_available());
    EXPECT_TRUE(dev_back.is_spillable());
    EXPECT_EQ(dev_back.sequence_number(), seq);
    EXPECT_EQ(dev_back.stream().value(), stream.value());
    EXPECT_EQ(dev_back.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_back.table_view(), expect);

    // Sanity check: a second device copy should also remain equivalent.
    auto dev_res2 = br->reserve_or_fail(
        dev_back.data_alloc_size(MemoryType::DEVICE), MemoryType::DEVICE
    );
    auto dev_copy2 = dev_back.copy(br.get(), dev_res2);
    EXPECT_TRUE(dev_copy2.is_available());
    EXPECT_EQ(dev_copy2.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_copy2.table_view(), expect);
}

TEST_F(StreamingTableChunk, ToMessageRoundTrip) {
    constexpr unsigned int num_rows = 64;
    constexpr std::int64_t seed = 2025;
    constexpr std::uint64_t seq = 7;

    auto expect = random_table_with_index(seed, num_rows, 0, 5);
    TableChunk chunk{seq, std::make_unique<cudf::table>(expect), stream};

    Message m = to_message(std::move(chunk));
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<TableChunk>());
    EXPECT_EQ(m.primary_data_size(MemoryType::HOST), std::make_pair(0, true));
    EXPECT_EQ(m.primary_data_size(MemoryType::DEVICE), std::make_pair(1024, true));

    // Deep-copy: device to host.
    auto reservation = br->reserve_or_fail(
        m.primary_data_size(MemoryType::DEVICE).first, MemoryType::HOST
    );
    Message m2 = m.copy(br.get(), reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m2.empty());
    EXPECT_TRUE(m2.holds<TableChunk>());
    EXPECT_EQ(m2.primary_data_size(MemoryType::HOST), std::make_pair(1024, true));
    EXPECT_EQ(m2.primary_data_size(MemoryType::DEVICE), std::make_pair(0, true));

    // Deep-copy: host to host.
    reservation = br->reserve_or_fail(
        m2.primary_data_size(MemoryType::HOST).first, MemoryType::HOST
    );
    Message m3 = m.copy(br.get(), reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m3.empty());
    EXPECT_TRUE(m3.holds<TableChunk>());
    EXPECT_EQ(m3.primary_data_size(MemoryType::HOST), std::make_pair(1024, true));
    EXPECT_EQ(m3.primary_data_size(MemoryType::DEVICE), std::make_pair(0, true));

    // Deep-copy: host to device.
    reservation = br->reserve_or_fail(
        m3.primary_data_size(MemoryType::HOST).first, MemoryType::DEVICE
    );
    Message m4 = m.copy(br.get(), reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m4.empty());
    EXPECT_TRUE(m4.holds<TableChunk>());
    EXPECT_EQ(m4.primary_data_size(MemoryType::HOST), std::make_pair(0, true));
    EXPECT_EQ(m4.primary_data_size(MemoryType::DEVICE), std::make_pair(1024, true));

    // Deep-copy: device to device.
    reservation = br->reserve_or_fail(
        m4.primary_data_size(MemoryType::DEVICE).first, MemoryType::DEVICE
    );
    Message m5 = m.copy(br.get(), reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m5.empty());
    EXPECT_TRUE(m5.holds<TableChunk>());
    EXPECT_EQ(m5.primary_data_size(MemoryType::HOST), std::make_pair(0, true));
    EXPECT_EQ(m5.primary_data_size(MemoryType::DEVICE), std::make_pair(1024, true));
}

TEST_F(StreamingTableChunk, ToMessageNotSpillable) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    auto deleter = [](void* p) { delete static_cast<int*>(p); };
    auto chunk = TableChunk{
        seq,
        expect,
        expect.alloc_size(),
        stream,
        OwningWrapper(new int, deleter),
        TableChunk::ExclusiveView::NO
    };

    Message m = to_message(std::move(chunk));
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<TableChunk>());
    EXPECT_EQ(m.primary_data_size(MemoryType::HOST), std::make_pair(0, false));
    EXPECT_EQ(
        m.primary_data_size(MemoryType::DEVICE),
        std::make_pair(expect.alloc_size(), false)
    );
}
