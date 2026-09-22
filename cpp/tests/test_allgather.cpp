/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/coll/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf::coll;

extern Environment* GlobalEnvironment;

class BaseAllGatherTest : public ::testing::Test {
  protected:
    void SetUp() override {
        stream = cuda::stream_ref{cudaStreamLegacy};
        br = rapidsmpf::BufferResource::create(rmm::mr::cuda_memory_resource{});
    }

    void TearDown() override {
        br = nullptr;
    }

    cuda::stream_ref stream{cudaStreamLegacy};
    std::shared_ptr<rapidsmpf::BufferResource> br;
};

TEST_F(BaseAllGatherTest, timeout) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    EXPECT_THROW(
        std::ignore = allgather.wait_and_extract(
            AllGather::Ordered::NO, std::chrono::milliseconds{20}
        ),
        std::runtime_error
    );
    allgather.insert_finished();
    std::vector<rapidsmpf::PackedData> result;
    EXPECT_NO_THROW(
        result =
            allgather.wait_and_extract(AllGather::Ordered::NO, std::chrono::seconds{30})
    );
    EXPECT_EQ(result.size(), 0);
}

class AllGatherTest
    : public BaseAllGatherTest,
      public ::testing::WithParamInterface<std::tuple<int, int, AllGather::Ordered>> {
  protected:
    void SetUp() override {
        BaseAllGatherTest::SetUp();
        std::tie(n_elements, n_inserts, ordered) = GetParam();
    }

    int n_elements;
    int n_inserts;
    AllGather::Ordered ordered;
};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGather,
    AllGatherTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 10, 100),  // n_elements
        ::testing::Values(0, 1, 10),  // n_inserts
        ::testing::Values(AllGather::Ordered::NO, AllGather::Ordered::YES)  // ordered
    ),
    [](const ::testing::TestParamInfo<AllGatherTest::ParamType>& info) {
        return "n_elements_" + std::to_string(std::get<0>(info.param)) + "n_inserts"
               + std::to_string(std::get<1>(info.param)) + "_"
               + (std::get<2>(info.param) == AllGather::Ordered::YES ? "ordered"
                                                                     : "unordered");
    }
);

constexpr auto gen_offset(int i, int r) {
    return i * 10 + r;
};

TEST_P(AllGatherTest, basic_allgather) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    auto const& comm = allgather.comm();
    auto this_rank = comm->rank();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        allgather.insert(i, std::move(packed_data));
    }

    allgather.insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    EXPECT_NO_THROW(
        results = allgather.wait_and_extract(ordered, std::chrono::seconds{30})
    );
    if (n_inserts > 0) {
        EXPECT_EQ(n_inserts * comm->nranks(), results.size());

        if (ordered == AllGather::Ordered::YES) {
            // results vector should be ordered by rank and insertion order. Values should
            // look like:
            // rank0    |0... |10...|... * n_inserts
            // rank1    |1... |11...|... * n_inserts
            // ...
            // rank n-1 |(n-1)... |...   * n_inserts
            for (int r = 0; r < comm->nranks(); r++) {
                for (int i = 0; i < n_inserts; i++) {
                    auto& result = results[r * n_inserts + i];
                    int exp_offset = gen_offset(i, r);
                    EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                        std::move(result), n_elements, exp_offset, stream, *br
                    ));
                }
            }
        } else {  // unordered
            std::vector<int> exp_offsets;
            for (int i = 0; i < n_inserts * comm->nranks(); i++) {
                exp_offsets.emplace_back(gen_offset(i % n_inserts, i / n_inserts));
            }

            for (auto&& result : results) {
                if (n_elements == 0) {
                    EXPECT_EQ(result.metadata->size(), 0);
                    continue;
                }
                int offset = *reinterpret_cast<int*>(result.metadata->data());
                auto it = std::ranges::find(exp_offsets, offset);
                EXPECT_NE(it, exp_offsets.end());
                exp_offsets.erase(it);
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), n_elements, offset, stream, *br
                ));
            }
            if (n_elements != 0) {
                EXPECT_TRUE(exp_offsets.empty());
            }
        }
    } else {  // n_inserts == 0. No data is inserted.
        EXPECT_EQ(0, results.size());
    }
}

TEST_F(BaseAllGatherTest, payload_statistics) {
    auto const& comm = GlobalEnvironment->comm_;
    ClearedStatistics statistics{comm->progress_thread()->statistics()};
    constexpr int n_elements = 7;
    constexpr int n_inserts = 3;

    AllGather allgather{comm, 0, br.get()};
    for (int i = 0; i < n_inserts; ++i) {
        allgather.insert(
            i, generate_packed_data(n_elements, gen_offset(i, comm->rank()), stream, *br)
        );
    }
    allgather.insert_finished();
    auto results =
        allgather.wait_and_extract(AllGather::Ordered::NO, std::chrono::seconds{30});
    EXPECT_EQ(results.size(), static_cast<std::size_t>(n_inserts * comm->nranks()));

    auto const expected_count =
        static_cast<std::size_t>(n_inserts * (comm->nranks() - 1));
    if (expected_count == 0) {
        EXPECT_THROW(statistics->get_stat("allgather-payload-send"), std::out_of_range);
        EXPECT_THROW(statistics->get_stat("allgather-payload-recv"), std::out_of_range);
    } else {
        auto const expected_message_size = n_elements * sizeof(int);
        auto const expected_bytes = expected_count * expected_message_size;
        auto const send = statistics->get_stat("allgather-payload-send");
        auto const recv = statistics->get_stat("allgather-payload-recv");
        EXPECT_EQ(send.count(), expected_count);
        EXPECT_EQ(send.value(), expected_bytes);
        EXPECT_EQ(send.max(), expected_message_size);
        EXPECT_EQ(recv.count(), expected_count);
        EXPECT_EQ(recv.value(), expected_bytes);
        EXPECT_EQ(recv.max(), expected_message_size);
    }
}

class AllGatherOrderedTest : public BaseAllGatherTest,
                             public ::testing::WithParamInterface<AllGather::Ordered> {};

// Parameterized test for different element counts
INSTANTIATE_TEST_SUITE_P(
    AllGatherOrdered,
    AllGatherOrderedTest,
    ::testing::Values(AllGather::Ordered::NO, AllGather::Ordered::YES),  // ordered,
    [](auto const& info) {
        return info.param == AllGather::Ordered::YES ? "ordered" : "unordered";
    }
);

TEST_P(AllGatherOrderedTest, allgatherv) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    auto const& comm = allgather.comm();
    auto ordered = GetParam();
    auto this_rank = comm->rank();
    constexpr int n_inserts = 4;
    auto n_ranks = comm->nranks();

    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(this_rank, gen_offset(i, this_rank), stream, *br);
        allgather.insert(i, std::move(packed_data));
    }

    allgather.insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    EXPECT_NO_THROW(
        results = allgather.wait_and_extract(ordered, std::chrono::seconds{30})
    );

    if (ordered == AllGather::Ordered::YES) {
        auto it = results.begin();
        for (int r = 0; r < n_ranks; r++) {
            for (int i = 0; i < n_inserts; i++) {
                auto& result = *it;
                EXPECT_EQ(r, static_cast<int>(result.metadata->size() / sizeof(int)));
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), r, gen_offset(i, r), stream, *br
                ));
                it++;
            }
        }
    } else {  // unordered
        for (auto&& result : results) {
            int n_elements = static_cast<int>(result.metadata->size() / sizeof(int));
            int offset =
                n_elements > 0 ? *reinterpret_cast<int*>(result.metadata->data()) : 0;
            EXPECT_NO_FATAL_FAILURE(
                validate_packed_data(std::move(result), n_elements, offset, stream, *br)
            );
        }
    }
}

TEST_P(AllGatherOrderedTest, non_uniform_inserts) {
    AllGather allgather{GlobalEnvironment->comm_, 0, br.get()};
    auto const& comm = allgather.comm();
    auto ordered = GetParam();
    auto this_rank = comm->rank();
    auto n_inserts = this_rank;
    auto n_ranks = comm->nranks();

    constexpr int n_elements = 5;

    // call insert this_rank times
    for (int i = 0; i < n_inserts; i++) {
        auto packed_data =
            generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br);
        allgather.insert(i, std::move(packed_data));
    }

    allgather.insert_finished();

    std::vector<rapidsmpf::PackedData> results;
    EXPECT_NO_THROW(
        results = allgather.wait_and_extract(ordered, std::chrono::seconds{30})
    );

    // results should be a triangular number of elements
    EXPECT_EQ((n_ranks - 1) * n_ranks / 2, results.size());

    if (ordered == AllGather::Ordered::YES) {
        auto it = results.begin();
        for (int r = 0; r < n_ranks; r++) {
            for (int i = 0; i < r; i++) {
                auto& result = *it;
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), n_elements, gen_offset(i, r), stream, *br
                ));
                it++;
            }
        }
    } else {  // unordered
        for (auto&& result : results) {
            if (result.data->size > 0) {
                int offset = *reinterpret_cast<int*>(result.metadata->data());
                EXPECT_NO_FATAL_FAILURE(validate_packed_data(
                    std::move(result), n_elements, offset, stream, *br
                ));
            }
        }
    }
}

// Test that reusing an OpID after a completed allgather doesn't cause cross-matching of
// messages between the old and new collective.
//
// On rank 0 we inject a stream-ordered delay into device allocations so that received
// chunks stay "not ready" in the event loop's to_receive_ queue. The event loop keeps
// running (the host is not blocked). With small messages, other ranks can post via eager
// protocols, complete, and move on to the next allgather. Its control messages will then
// be matched on rank 0 by the blocked previous allgather, unless we correctly stop
// polling once we've seen all control messages.
TEST_F(BaseAllGatherTest, opid_reuse) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() == 1) {
        GTEST_SKIP() << "OpID reuse test requires multiple ranks";
    }

    constexpr int n_elements = 10;
    constexpr int n_inserts = 2;
    auto this_rank = comm->rank();

    // On rank 0, wrap the device MR with a delayed version.
    std::shared_ptr<rapidsmpf::BufferResource> delay_br;
    std::unique_ptr<AllGather> allgather;
    constexpr rapidsmpf::OpID op_id = 0;
    if (this_rank == 0) {
        // Recreate the buffer resource and allgather with the delayed MR.
        delay_br = rapidsmpf::BufferResource::create(
            DelayedMemoryResource{br->device_mr(), std::chrono::milliseconds(500)}
        );
        allgather =
            std::make_unique<AllGather>(GlobalEnvironment->comm_, op_id, delay_br.get());
    } else {
        allgather =
            std::make_unique<AllGather>(GlobalEnvironment->comm_, op_id, br.get());
    }

    for (int i = 0; i < n_inserts; i++) {
        allgather->insert(
            i, generate_packed_data(n_elements, gen_offset(i, this_rank), stream, *br)
        );
    }

    allgather->insert_finished();
    std::vector<rapidsmpf::PackedData> results1;
    EXPECT_NO_THROW(
        results1 =
            allgather->wait_and_extract(AllGather::Ordered::YES, std::chrono::seconds{30})
    );
    // OK, it should be safe to reuse the opid now.
    allgather = std::make_unique<AllGather>(GlobalEnvironment->comm_, op_id, br.get());

    constexpr int second_offset = 1000;
    for (int i = 0; i < n_inserts; i++) {
        allgather->insert(
            i,
            generate_packed_data(
                n_elements, gen_offset(i + second_offset, this_rank), stream, *br
            )
        );
    }
    allgather->insert_finished();
    std::vector<rapidsmpf::PackedData> results2;
    EXPECT_NO_THROW(
        results2 =
            allgather->wait_and_extract(AllGather::Ordered::YES, std::chrono::seconds{30})
    );
    ASSERT_EQ(static_cast<std::size_t>(n_inserts * comm->nranks()), results1.size());
    for (auto&& result : results1) {
        int offset = *reinterpret_cast<int*>(result.metadata->data());
        EXPECT_NO_FATAL_FAILURE(
            validate_packed_data(std::move(result), n_elements, offset, stream, *br)
        );
    }

    ASSERT_EQ(static_cast<std::size_t>(n_inserts * comm->nranks()), results2.size());

    // Every result must carry data from the second allgather.
    for (auto&& result : results2) {
        int offset = *reinterpret_cast<int*>(result.metadata->data());
        EXPECT_GE(offset, second_offset);
        EXPECT_NO_FATAL_FAILURE(
            validate_packed_data(std::move(result), n_elements, offset, stream, *br)
        );
    }
}

// Test that PostBox::spill() tracks the remaining spill need correctly across iterations,
// rather than passing the original amount to lower_bound every time.
//
// With chunks [20, 80, 90] and a request for 100 bytes: the first iteration spills 90
// (the largest chunk, since none covers 100 alone). The second must search for a chunk
// >= 10 and pick the 20-byte chunk, totalling 110.
TEST(PostBox, spill_uses_remaining_amount) {
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    auto mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    auto br = rapidsmpf::BufferResource::create(*mr);

    rapidsmpf::coll::detail::PostBox postbox;

    auto make_chunk = [&](std::size_t size) {
        auto metadata =
            std::make_unique<std::vector<std::uint8_t>>(std::size_t{1}, std::uint8_t{0});
        auto res = br->reserve_or_fail(size, rapidsmpf::MemoryType::DEVICE);
        auto data = br->make_buffer(size, stream, res);
        return rapidsmpf::coll::detail::Chunk::from_packed_data(
            0,
            0,
            rapidsmpf::coll::detail::Chunk::INVALID_RANK,
            rapidsmpf::PackedData{std::move(metadata), std::move(data)}
        );
    };

    postbox.insert(make_chunk(20));
    postbox.insert(make_chunk(80));
    postbox.insert(make_chunk(90));

    EXPECT_EQ(postbox.spill(br.get(), 100, rapidsmpf::SPILL_TARGET_MEMORY_TYPES), 110UL);
}

namespace {

std::shared_ptr<rapidsmpf::BufferResource> make_allgather_disk_buffer_resource(
    TempDir const& temp_dir, std::int64_t device_limit, std::int64_t host_limit = 0
) {
    return rapidsmpf::BufferResource::create(
        rmm::mr::get_current_device_resource_ref(),
        rapidsmpf::PinnedMemoryDisabled,
        {{rapidsmpf::MemoryType::DEVICE, device_limit},
         {rapidsmpf::MemoryType::HOST, host_limit}},
        std::nullopt,
        std::make_shared<rapidsmpf::StreamPool>(4),
        rapidsmpf::Statistics::disabled(),
        temp_dir.path()
    );
}

std::unique_ptr<rapidsmpf::coll::detail::Chunk> make_allgather_disk_chunk(
    rapidsmpf::BufferResource& br,
    cuda::stream_ref stream,
    std::size_t n_elements,
    int offset
) {
    auto packed_data = generate_packed_data(n_elements, offset, stream, br);
    auto reservation =
        br.reserve_or_fail(packed_data.data->size, rapidsmpf::MemoryType::DISK);
    packed_data.data = br.move(std::move(packed_data.data), reservation);
    return rapidsmpf::coll::detail::Chunk::from_packed_data(
        0, 0, rapidsmpf::coll::detail::Chunk::INVALID_RANK, std::move(packed_data)
    );
}

}  // namespace

TEST(PostBox, spill_to_disk_preserves_payload) {
    constexpr std::size_t n_elements = 16;
    constexpr std::size_t data_size = n_elements * sizeof(int);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    TempDir temp_dir;
    auto br = make_allgather_disk_buffer_resource(temp_dir, data_size, 1LL << 40);
    rapidsmpf::coll::detail::PostBox postbox;
    postbox.insert(
        rapidsmpf::coll::detail::Chunk::from_packed_data(
            0,
            0,
            rapidsmpf::coll::detail::Chunk::INVALID_RANK,
            generate_packed_data(n_elements, 5, stream, *br)
        )
    );

    constexpr std::array spillable_memory_types{rapidsmpf::MemoryType::DISK};
    EXPECT_EQ(postbox.spill(br.get(), data_size, spillable_memory_types), data_size);

    auto chunks = postbox.extract();
    ASSERT_EQ(chunks.size(), 1);
    auto packed_data = chunks[0]->release();
    ASSERT_EQ(packed_data.data->mem_type(), rapidsmpf::MemoryType::DISK);
    auto const path =
        (*packed_data.data->get_storage<rapidsmpf::Buffer::DiskBufferT>()).path();
    EXPECT_TRUE(std::filesystem::exists(path));

    std::vector<rapidsmpf::PackedData> spilled;
    spilled.emplace_back(std::move(packed_data));
    auto restored = rapidsmpf::unspill_partitions(
        std::move(spilled), br.get(), rapidsmpf::AllowOverbooking::NO
    );
    ASSERT_EQ(restored.size(), 1);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(std::move(restored[0]), n_elements, 5, stream, *br)
    );
}

TEST(PostBox, extract_and_restore_restores_smallest_chunks_that_fit) {
    constexpr std::size_t largest_n_elements = 8;
    constexpr std::size_t addressable_limit = largest_n_elements * sizeof(int);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    TempDir temp_dir;
    auto br = make_allgather_disk_buffer_resource(
        temp_dir, addressable_limit, addressable_limit
    );
    rapidsmpf::coll::detail::PostBox postbox;
    for (auto const [n_elements, offset] : std::array<std::pair<std::size_t, int>, 3>{
             {{largest_n_elements, 80}, {2, 20}, {4, 40}}
         })
    {
        postbox.insert(make_allgather_disk_chunk(*br, stream, n_elements, offset));
    }
    stream.sync();

    constexpr std::array memory_types{rapidsmpf::MemoryType::HOST};
    auto ready = postbox.extract_and_restore(br.get(), memory_types);
    ASSERT_EQ(ready.size(), 2);
    EXPECT_EQ(postbox.size(), 1);
    std::ranges::sort(ready, std::less{}, [](auto const& chunk) {
        return chunk->data_size();
    });
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(ready[0]->release(), 2, 20, stream, *br)
    );
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(ready[1]->release(), 4, 40, stream, *br)
    );

    auto remaining = postbox.extract_and_restore(br.get(), memory_types);
    ASSERT_EQ(remaining.size(), 1);
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(remaining[0]->release(), largest_n_elements, 80, stream, *br)
    );
    EXPECT_TRUE(postbox.empty());
}

TEST(PostBox, extract_and_restore_reinserts_disk_chunks_when_reservation_fails) {
    constexpr std::size_t n_elements = 4;
    constexpr std::size_t data_size = n_elements * sizeof(int);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    TempDir temp_dir;
    auto br = make_allgather_disk_buffer_resource(temp_dir, data_size);
    rapidsmpf::coll::detail::PostBox postbox;
    postbox.insert(make_allgather_disk_chunk(*br, stream, n_elements, 5));
    stream.sync();

    constexpr std::array memory_types{rapidsmpf::MemoryType::HOST};
    EXPECT_THROW(
        std::ignore = postbox.extract_and_restore(br.get(), memory_types),
        std::runtime_error
    );
    EXPECT_EQ(postbox.size(), 1);

    br->set_memory_limit(rapidsmpf::MemoryType::HOST, data_size);
    auto restored = postbox.extract_and_restore(br.get(), memory_types);
    ASSERT_EQ(restored.size(), 1);
    EXPECT_TRUE(postbox.empty());
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(restored[0]->release(), n_elements, 5, stream, *br)
    );
}

TEST(PostBox, extract_and_restore_skips_not_ready_chunks) {
    constexpr std::size_t n_elements = 4;
    constexpr std::size_t data_size = n_elements * sizeof(int);
    auto stream = cuda::stream_ref{cudaStreamLegacy};
    TempDir temp_dir;
    auto br = make_allgather_disk_buffer_resource(temp_dir, data_size, 1LL << 40);

    auto disk_chunk = make_allgather_disk_chunk(*br, stream, n_elements, 5);
    auto disk_data = disk_chunk->release_data_buffer();
    auto const path = (*disk_data->get_storage<rapidsmpf::Buffer::DiskBufferT>()).path();
    disk_chunk->attach_data_buffer(std::move(disk_data));
    stream.sync();

    auto delayed_br = rapidsmpf::BufferResource::create(
        DelayedMemoryResource{br->device_mr(), std::chrono::milliseconds(500)}
    );
    auto delayed_stream = delayed_br->stream_pool()->get_stream();
    auto delayed_data = delayed_br->make_buffer(
        delayed_stream,
        delayed_br->reserve_or_fail(data_size, rapidsmpf::MemoryType::DEVICE)
    );
    auto delayed_chunk = rapidsmpf::coll::detail::Chunk::from_packed_data(
        0,
        0,
        rapidsmpf::coll::detail::Chunk::INVALID_RANK,
        rapidsmpf::PackedData{
            std::make_unique<std::vector<std::uint8_t>>(1, 0), std::move(delayed_data)
        }
    );
    ASSERT_FALSE(delayed_chunk->is_ready());

    rapidsmpf::coll::detail::PostBox postbox;
    postbox.insert(std::move(delayed_chunk));
    postbox.insert(std::move(disk_chunk));

    auto ready =
        postbox.extract_and_restore(br.get(), rapidsmpf::ADDRESSABLE_MEMORY_TYPES);
    ASSERT_EQ(ready.size(), 1);
    EXPECT_EQ(postbox.size(), 1);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(ready[0]->release(), n_elements, 5, stream, *br)
    );

    delayed_stream.sync();
    auto remaining =
        postbox.extract_and_restore(br.get(), rapidsmpf::ADDRESSABLE_MEMORY_TYPES);
    EXPECT_EQ(remaining.size(), 1);
    EXPECT_TRUE(postbox.empty());
}

TEST_F(BaseAllGatherTest, validates_configured_memory_types) {
    auto comm = GlobalEnvironment->split_comm();
    auto const addressable = rapidsmpf::to_vector(rapidsmpf::ADDRESSABLE_MEMORY_TYPES);

    EXPECT_THROW(
        AllGather(
            comm,
            0,
            br.get(),
            nullptr,
            std::vector{rapidsmpf::MemoryType::DEVICE},
            addressable
        ),
        std::invalid_argument
    );
    EXPECT_THROW(AllGather(comm, 0, br.get(), nullptr, {}, {}), std::invalid_argument);
    EXPECT_THROW(
        AllGather(
            comm, 0, br.get(), nullptr, {}, std::vector{rapidsmpf::MemoryType::DISK}
        ),
        std::invalid_argument
    );
}

TEST_F(BaseAllGatherTest, empty_spillable_memory_types_disable_spilling) {
    constexpr std::size_t n_elements = 16;
    constexpr std::size_t data_size = n_elements * sizeof(int);
    auto comm = GlobalEnvironment->split_comm();
    AllGather allgather(
        comm,
        0,
        br.get(),
        nullptr,
        {},
        rapidsmpf::to_vector(rapidsmpf::ADDRESSABLE_MEMORY_TYPES)
    );
    allgather.insert(0, generate_packed_data(n_elements, 5, stream, *br));
    EXPECT_EQ(br->spill_manager().spill(data_size), 0);
    allgather.insert_finished();

    auto results =
        allgather.wait_and_extract(AllGather::Ordered::YES, std::chrono::seconds{30});
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].data->mem_type(), rapidsmpf::MemoryType::DEVICE);
    EXPECT_NO_FATAL_FAILURE(
        validate_packed_data(std::move(results[0]), n_elements, 5, stream, *br)
    );
}

TEST_F(BaseAllGatherTest, disk_spill_round_trip) {
    constexpr std::int64_t force_spill_limit = -(1LL << 40);
    constexpr std::int64_t available_limit = 1LL << 40;
    constexpr std::size_t n_elements = 64;
    auto const& comm = GlobalEnvironment->comm_;
    auto const this_rank = comm->rank();
    TempDir temp_dir;
    auto disk_br =
        make_allgather_disk_buffer_resource(temp_dir, force_spill_limit, available_limit);
    AllGather allgather(
        comm,
        0,
        disk_br.get(),
        nullptr,
        {rapidsmpf::MemoryType::DISK},
        {rapidsmpf::MemoryType::HOST}
    );

    allgather.insert(0, generate_packed_data(n_elements, this_rank, stream, *disk_br));
    allgather.insert_finished();
    auto results =
        allgather.wait_and_extract(AllGather::Ordered::YES, std::chrono::seconds{30});
    ASSERT_EQ(results.size(), static_cast<std::size_t>(comm->nranks()));

    for (auto const& result : results) {
        EXPECT_EQ(
            result.data->mem_type(),
            comm->nranks() == 1 ? rapidsmpf::MemoryType::DISK
                                : rapidsmpf::MemoryType::HOST
        );
    }

    disk_br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, available_limit);
    auto restored = rapidsmpf::unspill_partitions(
        std::move(results), disk_br.get(), rapidsmpf::AllowOverbooking::NO
    );
    ASSERT_EQ(restored.size(), static_cast<std::size_t>(comm->nranks()));
    for (int rank = 0; rank < comm->nranks(); ++rank) {
        EXPECT_EQ(restored[rank].data->mem_type(), rapidsmpf::MemoryType::DEVICE);
        EXPECT_NO_FATAL_FAILURE(validate_packed_data(
            std::move(restored[rank]), n_elements, rank, stream, *disk_br
        ));
    }
}
