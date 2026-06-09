/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <ranges>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_actor.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace actor = rapidsmpf::streaming::actor;

class BaseStreamingShuffle : public BaseStreamingFixture {};

TEST_F(BaseStreamingShuffle, zero_owned_partitions_completes) {
    auto comm = GlobalEnvironment->comm_;
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Need at least 2 ranks so that some rank owns 0 partitions";
    }
    constexpr Rank owner = 0;
    auto collapse = [](std::shared_ptr<Communicator> const&,
                       shuffler::PartID,
                       shuffler::PartID) -> Rank { return owner; };
    constexpr OpID op_id = 0;
    constexpr shuffler::PartID total = 4;
    auto shuffler = std::make_unique<ShufflerAsync>(ctx, comm, op_id, total, collapse);

    coro::sync_wait(shuffler->insert_finished());

    auto local_pids = shuffler->local_partitions();
    if (comm->rank() == owner) {
        EXPECT_EQ(local_pids.size(), total);
    } else {
        EXPECT_TRUE(local_pids.empty());
    }
}

class StreamingShuffler : public BaseStreamingShuffle,
                          public ::testing::WithParamInterface<int> {
  public:
    static constexpr size_t num_partitions = 10;
    static constexpr size_t num_rows = 1000;
    static constexpr size_t num_chunks = 5;
    static constexpr OpID op_id = 0;

    shuffler::Shuffler::PartitionOwner partition_owner = shuffler::Shuffler::round_robin;

    void SetUp() override {
        BaseStreamingShuffle::SetUpWithThreads(GetParam());
    }

    void TearDown() override {
        BaseStreamingShuffle::TearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(
    StreamingShuffler,
    StreamingShuffler,
    ::testing::Values(1, 2, 4),
    [](testing::TestParamInfo<StreamingShuffler::ParamType> const& info) {
        return "nthreads_" + std::to_string(info.param);
    }
);

// Verifies end-to-end correctness of the streaming shuffler actor.
//
// Each rank sends num_chunks messages, where each message is a PartitionMapChunk covering
// all num_partitions partitions. The data for partition j in chunk c_idx on rank r is a
// contiguous integer sequence whose values encode both the rank and the row range, making
// them globally unique and independently verifiable.
//
// After shuffling, each local partition should have received exactly num_chunks * nranks
// packed-data items (one per (rank, chunk) pair). The items are sorted by their first
// element — which equals rank * num_rows + row_start — to reconstruct rank-major,
// chunk-minor order, and then validated against the expected row range from `pieces`.
TEST_P(StreamingShuffler, basic_shuffler) {
    auto comm = GlobalEnvironment->comm_;
    // split a span [0, num_rows) into num_partitions * num_chunks contiguous pieces.
    auto const pieces = rapidsmpf::chunk_indices(num_rows, num_partitions * num_chunks);

    // each rank contributes num_chunks messages, each covering num_partitions pieces.
    const int64_t base = static_cast<int64_t>(comm->rank()) * num_rows;
    std::vector<Message> input_chunks;  // Message contains a PartitionMapChunk
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        std::unordered_map<shuffler::PartID, PackedData> chunks;
        chunks.reserve(num_partitions);
        for (size_t j = 0; j < num_partitions; ++j) {
            auto [start, end] = pieces[chunk_idx * num_partitions + j];
            // end > start is guaranteed.
            chunks.emplace(
                static_cast<shuffler::PartID>(j),
                generate_packed_data<int64_t>(
                    end - start, base + static_cast<int64_t>(start), stream, *br
                )
            );
        }
        input_chunks.emplace_back(
            to_message(chunk_idx, std::make_unique<PartitionMapChunk>(std::move(chunks)))
        );
    }
    EXPECT_EQ(input_chunks.size(), num_chunks);

    // Create and run the streaming pipeline.
    std::vector<Message> output_chunks;
    {
        std::vector<Actor> actors;
        auto ch1 = ctx->create_channel();
        actors.push_back(actor::push_to_channel(ctx, ch1, std::move(input_chunks)));

        auto ch2 = ctx->create_channel();
        actors.emplace_back(
            actor::shuffler(ctx, comm, ch1, ch2, op_id, num_partitions, partition_owner)
        );

        actors.push_back(actor::pull_from_channel(ctx, ch2, output_chunks));

        run_actor_network(std::move(actors));
    }

    auto local_pids =
        shuffler::Shuffler::local_partitions(comm, num_partitions, partition_owner);

    // Since all partitions are non-empty, each local partition ID should a corresponding
    // output chunk.
    EXPECT_EQ(local_pids.size(), output_chunks.size());
    const size_t n_ranks = static_cast<size_t>(comm->nranks());
    for (auto& chunk : output_chunks) {
        auto pid = chunk.sequence_number();
        std::erase_if(local_pids, [pid](auto& p) { return p == pid; });

        auto p_vec = chunk.release<PartitionVectorChunk>();
        // for each local pid, it should receive num_chunks * nranks chunks.
        ASSERT_EQ(p_vec.data.size(), num_chunks * n_ranks);

        // since values are offset by rank, if we sort packed data by their first element,
        // then it will be in rank & chunk-index order.
        std::ranges::sort(p_vec.data, [](auto& a, auto& b) {
            std::int64_t oa{}, ob{};
            std::memcpy(&oa, a.metadata->data(), sizeof(std::int64_t));
            std::memcpy(&ob, b.metadata->data(), sizeof(std::int64_t));
            return oa < ob;
        });

        // p_vec.data is sorted by first element, so entries are ordered
        // (rank=0,chunk=0), (rank=0,chunk=1), ..., (rank=1,chunk=0), ...
        // i.e. flat index r*num_chunks + c_idx.
        for (size_t r = 0; r < n_ranks; ++r) {
            auto r_base = static_cast<int64_t>(r) * num_rows;  // rank-base offset
            for (size_t c_idx = 0; c_idx < num_chunks; ++c_idx) {
                const auto [start, end] = pieces[c_idx * num_partitions + pid];
                SCOPED_TRACE(
                    "pid=" + std::to_string(pid) + ", rank=" + std::to_string(r)
                    + ", chunk_idx=" + std::to_string(c_idx)
                );
                validate_packed_data<int64_t>(
                    std::move(p_vec.data[r * num_chunks + c_idx]),
                    end - start,
                    r_base + static_cast<int64_t>(start),
                    stream,
                    *br
                );
            }
        }
    }
    EXPECT_TRUE(local_pids.empty());
}

class ShufflerAsyncTest
    : public BaseStreamingShuffle,
      public ::testing::WithParamInterface<std::tuple<std::size_t, std::uint32_t>> {
  protected:
    std::size_t n_inserts;
    std::uint32_t n_partitions;

    static constexpr OpID op_id = 0;
    static constexpr std::size_t n_elements = 100;

    void SetUp() override {
        std::tie(n_inserts, n_partitions) = GetParam();

        BaseStreamingShuffle::SetUpWithThreads(4);
    }

    void TearDown() override {
        BaseStreamingShuffle::TearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(
    StreamingShuffler,
    ShufflerAsyncTest,
    ::testing::Combine(
        ::testing::Values(1, 10),  // number of inserts
        ::testing::Values(1, 10, 100)  // number of partitions
    ),
    [](const testing::TestParamInfo<ShufflerAsyncTest::ParamType>& info) {
        return "ninserts_" + std::to_string(std::get<0>(info.param)) + "_nparts_"
               + std::to_string(std::get<1>(info.param));
    }
);

TEST_P(ShufflerAsyncTest, insert_wait_extract) {
    auto comm = GlobalEnvironment->comm_;
    auto shuffler = std::make_unique<ShufflerAsync>(ctx, comm, op_id, n_partitions);

    for (std::size_t i = 0; i < n_inserts; ++i) {
        std::unordered_map<shuffler::PartID, PackedData> data;
        data.reserve(n_partitions);
        for (shuffler::PartID pid = 0; pid < n_partitions; ++pid) {
            data.emplace(pid, generate_packed_data(n_elements, 0, stream, *br));
        }
        shuffler->insert(std::move(data));
    }

    coro::sync_wait(shuffler->insert_finished());

    auto local_pids = shuffler::Shuffler::local_partitions(
        comm, n_partitions, &shuffler::Shuffler::round_robin
    );

    std::vector<shuffler::PartID> finished_pids;
    std::size_t n_chunks_received = 0;
    for (auto pid : local_pids) {
        auto chunks = shuffler->extract(pid);
        n_chunks_received += chunks.size();
        finished_pids.push_back(pid);
    }

    EXPECT_EQ(n_inserts * local_pids.size() * comm->nranks(), n_chunks_received);
    EXPECT_EQ(local_pids, finished_pids);
}
