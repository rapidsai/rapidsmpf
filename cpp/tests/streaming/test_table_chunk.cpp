/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"


using namespace rapidsmpf;
using namespace rapidsmpf::streaming;

class StreamingTableChunk : public BaseStreamingFixture,
                            public ::testing::WithParamInterface<rapidsmpf::MemoryType> {
  protected:
    void SetUp() override {
        rapidsmpf::config::Options options(
            rapidsmpf::config::get_environment_variables()
        );

        std::unordered_map<MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
            memory_available{};
        auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(
            16, rmm::cuda_stream::flags::non_blocking
        );
        stream = cudf::get_default_stream();
        br = std::make_shared<rapidsmpf::BufferResource>(
            mr_cuda,  // device_mr
            rapidsmpf::PinnedMemoryResource::make_if_available(),  // pinned_mr
            memory_available,  // memory_available
            std::chrono::milliseconds{1},  // periodic_spill_check
            stream_pool,  // stream_pool
            Statistics::disabled()  // statistics
        );
        ctx = std::make_shared<rapidsmpf::streaming::Context>(
            options, GlobalEnvironment->comm_, br
        );
    }

    rmm::cuda_stream_view stream;
    rmm::mr::cuda_memory_resource mr_cuda;
    std::shared_ptr<rapidsmpf::BufferResource> br;
    std::shared_ptr<rapidsmpf::streaming::Context> ctx;
};

TEST_F(StreamingTableChunk, FromTable) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    TableChunk chunk{std::make_unique<cudf::table>(expect), stream};
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
            expect, stream, OwningWrapper(new int, deleter), exclusive_view
        };
    };
    auto check_chunk = [&](TableChunk const& chunk, bool is_spillable) {
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
        auto msg = to_message(
            seq, std::make_unique<TableChunk>(make_chunk(TableChunk::ExclusiveView::NO))
        );
        EXPECT_EQ(num_deletions, 1);
    }
    EXPECT_EQ(num_deletions, 2);
    {
        auto msg = to_message(
            seq, std::make_unique<TableChunk>(make_chunk(TableChunk::ExclusiveView::YES))
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
        chunk = chunk.copy(res);
        EXPECT_EQ(num_deletions, 4);
    }
}

TEST_F(StreamingTableChunk, FromPackedColumns) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed = cudf::pack(expect, stream);

    TableChunk chunk{std::make_unique<cudf::packed_columns>(std::move(packed)), stream};

    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_TRUE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_EQ(chunk.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
}

TEST_F(StreamingTableChunk, FromPackedDataOnDevice) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed_columns = cudf::pack(expect, stream);

    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        br->move(std::move(packed_columns.gpu_data), stream)
    );
    TableChunk chunk{std::move(packed_data)};

    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_FALSE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_THROW((void)chunk.table_view(), std::invalid_argument);

    // Eventhough the table isn't available, it is still all in device memory
    // so the memory cost of making it available is zero.
    EXPECT_EQ(chunk.make_available_cost(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    StreamingTableChunkWithSpillTargets,
    StreamingTableChunk,
    ::testing::ValuesIn(rapidsmpf::SPILL_TARGET_MEMORY_TYPES),
    [](testing::TestParamInfo<rapidsmpf::MemoryType> const& info) {
        return std::string{rapidsmpf::to_string(info.param)};
    }
);

TEST_P(StreamingTableChunk, FromPackedDataOn) {
    auto const spill_mem_type = GetParam();
    if (spill_mem_type == MemoryType::PINNED_HOST
        && !is_pinned_memory_resources_supported())
    {
        GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
    }

    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);
    auto packed_columns = cudf::pack(expect, stream);
    std::size_t const size = packed_columns.gpu_data->size();

    // Move the gpu_data to a Buffer (still device memory).
    auto gpu_data_on_device = br->move(std::move(packed_columns.gpu_data), stream);

    // Copy the GPU data to the current spill target memory type.
    auto [res, _] = br->reserve(spill_mem_type, size, true);
    auto gpu_data_in_spill_memory = br->move(std::move(gpu_data_on_device), res);

    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata), std::move(gpu_data_in_spill_memory)
    );
    TableChunk chunk{std::move(packed_data)};

    EXPECT_EQ(chunk.stream().value(), stream.value());
    EXPECT_FALSE(chunk.is_available());
    EXPECT_TRUE(chunk.is_spillable());
    EXPECT_THROW((void)chunk.table_view(), std::invalid_argument);
    EXPECT_EQ(chunk.make_available_cost(), size);
}

TEST_F(StreamingTableChunk, DeviceToDeviceCopy) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;

    auto expect = random_table_with_index(seed, num_rows, 0, 10);

    rapidsmpf::streaming::TableChunk chunk{std::make_unique<cudf::table>(expect), stream};
    EXPECT_TRUE(chunk.is_available());

    auto res = br->reserve_or_fail(
        chunk.data_alloc_size(MemoryType::DEVICE), MemoryType::DEVICE
    );
    auto chunk2 = chunk.copy(res);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk2.table_view(), expect);
}

TEST_P(StreamingTableChunk, DeviceToHostRoundTripCopy) {
    auto const spill_mem_type = GetParam();
    if (spill_mem_type == MemoryType::PINNED_HOST
        && !is_pinned_memory_resources_supported())
    {
        GTEST_SKIP() << "MemoryType::PINNED_HOST isn't supported on the system.";
    }

    constexpr unsigned int num_rows = 64;
    constexpr std::int64_t seed = 2025;

    auto expect = random_table_with_index(seed, num_rows, 0, 5);

    TableChunk dev_chunk{std::make_unique<cudf::table>(expect), stream};
    EXPECT_TRUE(dev_chunk.is_available());
    EXPECT_TRUE(dev_chunk.is_spillable());
    EXPECT_EQ(dev_chunk.stream().value(), stream.value());
    EXPECT_EQ(dev_chunk.make_available_cost(), 0);
    {
        auto cd = get_content_description(dev_chunk);
        EXPECT_EQ(cd.spillable(), dev_chunk.is_spillable());
        for (auto mem_type : MEMORY_TYPES) {
            EXPECT_EQ(cd.content_size(mem_type), dev_chunk.data_alloc_size(mem_type));
        }
    }

    // Copy to host memory -> new chunk should be unavailable.
    auto host_res = br->reserve_or_fail(
        dev_chunk.data_alloc_size(MemoryType::DEVICE), spill_mem_type
    );
    auto host_copy = dev_chunk.copy(host_res);
    EXPECT_FALSE(host_copy.is_available());
    EXPECT_TRUE(host_copy.is_spillable());
    EXPECT_EQ(host_copy.stream().value(), stream.value());
    EXPECT_GT(host_copy.make_available_cost(), 0);
    {
        auto cd = get_content_description(host_copy);
        EXPECT_EQ(cd.spillable(), host_copy.is_spillable());
        for (auto mem_type : MEMORY_TYPES) {
            EXPECT_EQ(cd.content_size(mem_type), host_copy.data_alloc_size(mem_type));
        }
    }

    // Host to host copy.
    auto host_res2 =
        br->reserve_or_fail(host_copy.data_alloc_size(spill_mem_type), spill_mem_type);
    auto host_copy2 = host_copy.copy(host_res2);
    EXPECT_FALSE(host_copy2.is_available());
    EXPECT_TRUE(host_copy2.is_spillable());
    EXPECT_EQ(host_copy2.stream().value(), stream.value());
    EXPECT_EQ(host_copy2.make_available_cost(), host_copy.make_available_cost());
    {
        auto cd = get_content_description(host_copy2);
        EXPECT_EQ(cd.spillable(), host_copy2.is_spillable());
        for (auto mem_type : MEMORY_TYPES) {
            EXPECT_EQ(cd.content_size(mem_type), host_copy2.data_alloc_size(mem_type));
        }
    }

    // Bring the new host copy back to device and verify equality.
    auto dev_res = br->reserve_or_fail(
        host_copy2.data_alloc_size(spill_mem_type), MemoryType::DEVICE
    );
    auto dev_back = host_copy2.make_available(dev_res);
    EXPECT_TRUE(dev_back.is_available());
    EXPECT_TRUE(dev_back.is_spillable());
    EXPECT_EQ(dev_back.stream().value(), stream.value());
    EXPECT_EQ(dev_back.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_back.table_view(), expect);
    {
        auto cd = get_content_description(dev_back);
        EXPECT_EQ(cd.spillable(), dev_back.is_spillable());
        for (auto mem_type : MEMORY_TYPES) {
            EXPECT_EQ(cd.content_size(mem_type), dev_back.data_alloc_size(mem_type));
        }
    }

    // Sanity check: a second device copy should also remain equivalent.
    auto dev_res2 = br->reserve_or_fail(
        dev_back.data_alloc_size(MemoryType::DEVICE), MemoryType::DEVICE
    );
    auto dev_copy2 = dev_back.copy(dev_res2);
    EXPECT_TRUE(dev_copy2.is_available());
    EXPECT_EQ(dev_copy2.make_available_cost(), 0);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(dev_copy2.table_view(), expect);
    {
        auto cd = get_content_description(dev_copy2);
        EXPECT_EQ(cd.spillable(), dev_copy2.is_spillable());
        for (auto mem_type : MEMORY_TYPES) {
            EXPECT_EQ(cd.content_size(mem_type), dev_copy2.data_alloc_size(mem_type));
        }
    }
}

TEST_F(StreamingTableChunk, ToMessageRoundTrip) {
    constexpr unsigned int num_rows = 64;
    constexpr std::int64_t seed = 2025;
    constexpr std::uint64_t seq = 7;

    auto expect = random_table_with_index(seed, num_rows, 0, 5);
    auto chunk =
        std::make_unique<TableChunk>(std::make_unique<cudf::table>(expect), stream);

    Message m = to_message(seq, std::move(chunk));
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<TableChunk>());
    EXPECT_TRUE(m.content_description().spillable());
    EXPECT_EQ(m.content_description().content_size(MemoryType::HOST), 0);
    EXPECT_EQ(m.content_description().content_size(MemoryType::DEVICE), 1024);
    EXPECT_EQ(m.sequence_number(), seq);

    // Deep-copy: device to host.
    auto reservation = br->reserve_or_fail(m.copy_cost(), MemoryType::HOST);
    Message m2 = m.copy(reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m2.empty());
    EXPECT_TRUE(m2.holds<TableChunk>());
    EXPECT_TRUE(m2.content_description().spillable());
    EXPECT_EQ(m2.content_description().content_size(MemoryType::HOST), 1024);
    EXPECT_EQ(m2.content_description().content_size(MemoryType::DEVICE), 0);
    EXPECT_EQ(m2.sequence_number(), seq);

    // Deep-copy: host to host.
    reservation = br->reserve_or_fail(m2.copy_cost(), MemoryType::HOST);
    Message m3 = m.copy(reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m3.empty());
    EXPECT_TRUE(m3.holds<TableChunk>());
    EXPECT_TRUE(m3.content_description().spillable());
    EXPECT_EQ(m3.content_description().content_size(MemoryType::HOST), 1024);
    EXPECT_EQ(m3.content_description().content_size(MemoryType::DEVICE), 0);
    EXPECT_EQ(m3.sequence_number(), seq);

    // Copy the chunk back to device and verify.
    {
        auto chunk = m3.release<TableChunk>();
        auto res = br->reserve_or_fail(chunk.make_available_cost(), MemoryType::DEVICE);
        chunk = chunk.make_available(res);
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(chunk.table_view(), expect);
    }

    // Deep-copy: host to device.
    reservation = br->reserve_or_fail(m2.copy_cost(), MemoryType::DEVICE);
    Message m4 = m.copy(reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m4.empty());
    EXPECT_TRUE(m4.holds<TableChunk>());
    EXPECT_TRUE(m4.content_description().spillable());
    EXPECT_EQ(m4.content_description().content_size(MemoryType::HOST), 0);
    EXPECT_EQ(m4.content_description().content_size(MemoryType::DEVICE), 1024);
    EXPECT_EQ(m4.sequence_number(), seq);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(m4.get<TableChunk>().table_view(), expect);

    // Deep-copy: device to device.
    reservation = br->reserve_or_fail(m4.copy_cost(), MemoryType::DEVICE);
    Message m5 = m.copy(reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m5.empty());
    EXPECT_TRUE(m5.holds<TableChunk>());
    EXPECT_TRUE(m5.content_description().spillable());
    EXPECT_EQ(m5.content_description().content_size(MemoryType::HOST), 0);
    EXPECT_EQ(m5.content_description().content_size(MemoryType::DEVICE), 1024);
    EXPECT_EQ(m5.sequence_number(), seq);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(m5.get<TableChunk>().table_view(), expect);
}

TEST_F(StreamingTableChunk, ToMessageNotSpillable) {
    constexpr unsigned int num_rows = 100;
    constexpr std::int64_t seed = 1337;
    constexpr std::uint64_t seq = 42;

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    auto deleter = [](void* p) { delete static_cast<int*>(p); };
    auto chunk = std::make_unique<TableChunk>(
        expect, stream, OwningWrapper(new int, deleter), TableChunk::ExclusiveView::NO
    );

    Message m = to_message(seq, std::move(chunk));
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<TableChunk>());
    EXPECT_FALSE(m.content_description().spillable());
    EXPECT_EQ(m.content_description().content_size(MemoryType::HOST), 0);
    EXPECT_EQ(
        m.content_description().content_size(MemoryType::DEVICE), expect.alloc_size()
    );
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(m.get<TableChunk>().table_view(), expect);
}

TEST_F(StreamingTableChunk, ToMessageUnalignedSize) {
    constexpr unsigned int num_rows = 5;
    constexpr std::int64_t seed = 2025;
    constexpr std::uint64_t seq = 7;

    auto expect = random_table_with_index(seed, num_rows, 0, 5);
    auto chunk =
        std::make_unique<TableChunk>(std::make_unique<cudf::table>(expect), stream);

    Message m = to_message(seq, std::move(chunk));
    EXPECT_EQ(m.sequence_number(), seq);
    EXPECT_FALSE(m.empty());
    EXPECT_TRUE(m.holds<TableChunk>());
    EXPECT_TRUE(m.content_description().spillable());
    EXPECT_EQ(m.content_description().content_size(MemoryType::HOST), 0);
    EXPECT_EQ(m.content_description().content_size(MemoryType::DEVICE), 80);
    EXPECT_EQ(m.copy_cost(), 80);

    // Deep copy: device → host.
    // Note: `m.copy_cost() == 80`, but cudf performs 128-byte aligned allocations.
    // This means `m.copy_cost()` is not always sufficient; however, TableChunk.copy()
    // accounts for this alignment internally.
    auto reservation = br->reserve_or_fail(m.copy_cost(), MemoryType::HOST);
    Message m2 = m.copy(reservation);
    EXPECT_EQ(reservation.size(), 0);
    EXPECT_FALSE(m2.empty());
    EXPECT_TRUE(m2.holds<TableChunk>());
    EXPECT_TRUE(m2.content_description().spillable());
    EXPECT_EQ(m2.copy_cost(), 128);
    EXPECT_EQ(m2.content_description().content_size(MemoryType::HOST), 128);
    EXPECT_EQ(m2.content_description().content_size(MemoryType::DEVICE), 0);
    EXPECT_EQ(m2.sequence_number(), seq);
}
