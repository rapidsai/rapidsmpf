/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

class BaseStreamingShuffle : public BaseStreamingFixture {};

class StreamingShuffler : public BaseStreamingShuffle,
                          public ::testing::WithParamInterface<int> {
  public:
    const unsigned int num_partitions = 10;
    const unsigned int num_rows = 1000;
    const unsigned int num_chunks = 5;
    const unsigned int chunk_size = num_rows / num_chunks;
    const std::int64_t seed = 42;
    const cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;
    const OpID op_id = 0;

    void SetUp() override {
        BaseStreamingShuffle::SetUpWithThreads(GetParam());
        GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
        BaseStreamingShuffle::TearDown();
    }

    void run_test(auto make_shuffler_node_fn) {
        // Create the full input table and slice it into chunks.
        cudf::table full_input_table = random_table_with_index(seed, num_rows, 0, 10);
        std::vector<Message> input_chunks;
        for (unsigned int i = 0; i < num_chunks; ++i) {
            input_chunks.emplace_back(to_message(
                i,
                std::make_unique<TableChunk>(
                    std::make_unique<cudf::table>(
                        cudf::slice(
                            full_input_table,
                            {static_cast<cudf::size_type>(i * chunk_size),
                             static_cast<cudf::size_type>((i + 1) * chunk_size)},
                            stream
                        )
                            .at(0),
                        stream,
                        ctx->br()->device_mr()
                    ),
                    stream
                )
            ));
        }

        // Create and run the streaming pipeline.
        std::vector<Message> output_chunks;
        {
            std::vector<Node> nodes;
            auto ch1 = ctx->create_channel();
            nodes.push_back(node::push_to_channel(ctx, ch1, std::move(input_chunks)));

            auto ch2 = ctx->create_channel();
            nodes.push_back(
                node::partition_and_pack(
                    ctx, ch1, ch2, {1}, num_partitions, hash_function, seed
                )
            );

            auto ch3 = ctx->create_channel();
            nodes.emplace_back(make_shuffler_node_fn(ch2, ch3));

            auto ch4 = ctx->create_channel();
            nodes.push_back(node::unpack_and_concat(ctx, ch3, ch4));

            nodes.push_back(node::pull_from_channel(ctx, ch4, output_chunks));

            run_streaming_pipeline(std::move(nodes));
        }

        std::unique_ptr<cudf::table> expected_table;
        if (ctx->comm()->nranks() == 1) {  // full_input table is expected
            expected_table = std::make_unique<cudf::table>(std::move(full_input_table));
        } else {  // full_input table is replicated on all ranks
            // local partitions
            auto [table, offsets] = cudf::hash_partition(
                full_input_table.view(), {1}, num_partitions, hash_function, seed
            );

            auto local_pids = shuffler::Shuffler::local_partitions(
                ctx->comm(), num_partitions, shuffler::Shuffler::round_robin
            );

            // every partition is replicated on all ranks
            std::vector<cudf::table_view> expected_tables;
            for (auto pid : local_pids) {
                auto t_view =
                    cudf::slice(table->view(), {offsets[pid], offsets[pid + 1]}).at(0);
                // this will be replicated on all ranks
                for (rapidsmpf::Rank rank = 0; rank < ctx->comm()->nranks(); ++rank) {
                    expected_tables.push_back(t_view);
                }
            }
            expected_table = cudf::concatenate(expected_tables);
        }

        // Concat all output chunks to a single table.
        std::vector<cudf::table_view> output_chunks_as_views;
        for (auto& chunk : output_chunks) {
            output_chunks_as_views.push_back(chunk.get<TableChunk>().table_view());
        }
        auto result_table = cudf::concatenate(output_chunks_as_views);

        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(result_table->view()), sort_table(expected_table->view())
        );
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

TEST_P(StreamingShuffler, basic_shuffler) {
    EXPECT_NO_FATAL_FAILURE(run_test([&](auto ch_in, auto ch_out) -> Node {
        return node::shuffler(ctx, ch_in, ch_out, op_id, num_partitions);
    }));
}

class ShufflerAsyncTest
    : public BaseStreamingShuffle,
      public ::testing::WithParamInterface<std::tuple<int, size_t, uint32_t, int>> {
  protected:
    int n_threads;
    size_t n_inserts;
    uint32_t n_partitions;
    int n_consumers;

    static constexpr OpID op_id = 0;
    static constexpr size_t n_elements = 100;

    void SetUp() override {
        std::tie(n_threads, n_inserts, n_partitions, n_consumers) = GetParam();

        BaseStreamingShuffle::SetUpWithThreads(n_threads);
        GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
        BaseStreamingShuffle::TearDown();
    }
};

INSTANTIATE_TEST_SUITE_P(
    StreamingShuffler,
    ShufflerAsyncTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4),  // number of streaming threads
        ::testing::Values(1, 10),  // number of inserts
        ::testing::Values(1, 10, 100),  // number of partitions
        ::testing::Values(1, 4)  // number of consumers
    ),
    [](const testing::TestParamInfo<ShufflerAsyncTest::ParamType>& info) {
        return "nthreads_" + std::to_string(std::get<0>(info.param)) + "_ninserts_"
               + std::to_string(std::get<1>(info.param)) + "_nparts_"
               + std::to_string(std::get<2>(info.param)) + "_nconsumers_"
               + std::to_string(std::get<3>(info.param));
    }
);

TEST_P(ShufflerAsyncTest, multi_consumer_extract) {
    auto shuffler = std::make_unique<ShufflerAsync>(ctx, op_id, n_partitions);
    // extract data (executed by thread pool)
    auto extract_task = [](int tid,
                           auto* shuffler,
                           auto* ctx,
                           std::mutex& mtx,
                           std::vector<shuffler::PartID>& finished_pids,
                           size_t& n_chunks_received) -> Node {
        co_await ctx->executor()->schedule();
        ctx->comm()->logger().debug(tid, " extract task started");

        while (true) {
            auto result = co_await shuffler->extract_any_async();
            if (!result.has_value()) {
                break;
            }
            auto lock = std::unique_lock(mtx);
            auto& [pid, chunks] = *result;
            n_chunks_received += chunks.size();
            finished_pids.push_back(pid);
        }
        ctx->comm()->logger().debug(tid, " extract task finished");
    };

    for (size_t i = 0; i < n_inserts; ++i) {
        std::unordered_map<shuffler::PartID, PackedData> data;
        data.reserve(n_partitions);
        for (shuffler::PartID pid = 0; pid < n_partitions; ++pid) {
            data.emplace(pid, generate_packed_data(n_elements, 0, stream, *br));
        }
        shuffler->insert(std::move(data));
    }

    auto finish_token = shuffler->insert_finished();

    std::mutex mtx;
    std::vector<shuffler::PartID> finished_pids;
    size_t n_chunks_received = 0;
    std::vector<Node> tasks;
    for (int i = 0; i < n_consumers; ++i) {
        tasks.emplace_back(extract_task(
            i, shuffler.get(), ctx.get(), mtx, finished_pids, n_chunks_received
        ));
    }
    tasks.push_back(ctx->executor()->schedule(std::move(finish_token)));
    run_streaming_pipeline(std::move(tasks));

    auto local_pids = shuffler::Shuffler::local_partitions(
        ctx->comm(), n_partitions, shuffler::Shuffler::round_robin
    );
    EXPECT_EQ(n_inserts * local_pids.size() * ctx->comm()->nranks(), n_chunks_received);

    std::ranges::sort(finished_pids);
    EXPECT_EQ(local_pids, finished_pids);
}

TEST_F(BaseStreamingShuffle, extract_any_before_extract) {
    GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
    static constexpr OpID op_id = 0;
    static constexpr size_t n_partitions = 10;
    {
        auto shuffler = std::make_unique<ShufflerAsync>(ctx, op_id, n_partitions);

        // all empty partitions
        auto finish_token = shuffler->insert_finished();

        auto local_pids = shuffler::Shuffler::local_partitions(
            ctx->comm(), n_partitions, shuffler::Shuffler::round_robin
        );

        size_t parts_extracted = 0;
        // For this test we need to await the shuffler being finished and drained, i.e.
        // ensure all insertion notifications have been received before extracting. This
        // is only because we sync_wait each individual extract_any_async.
        coro::sync_wait(finish_token);
        while (true) {  // extract all partitions
            if (!coro::sync_wait(shuffler->extract_any_async()).has_value()) {
                break;
            }
            parts_extracted++;
        }
        EXPECT_EQ(local_pids.size(), parts_extracted);
        // now extract should return std::nullopt.
        for (auto pid : local_pids) {
            EXPECT_EQ(coro::sync_wait(shuffler->extract_async(pid)), std::nullopt);
        }
    }
    GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
}

class CompetingShufflerAsyncTest : public BaseStreamingShuffle {
  public:
    void SetUp() override {
        BaseStreamingShuffle::SetUp();
        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
        BaseStreamingShuffle::TearDown();
    }

  protected:
    // produce_results_fn is a function that produces the results of the extract_any_async
    // and extract_async coroutines.
    void run_test(auto produce_results_fn) {
        static constexpr OpID op_id = 0;
        shuffler::PartID const n_partitions = ctx->comm()->nranks();
        shuffler::PartID const this_pid = ctx->comm()->rank();

        auto shuffler = std::make_unique<ShufflerAsync>(ctx, op_id, n_partitions);

        auto finish_token = shuffler->insert_finished();
        coro::sync_wait(finish_token);
        auto [extract_any_result, extract_result] =
            produce_results_fn(shuffler.get(), this_pid);

        // if extract_any_result is valid, then extract_result should return nullopt
        if (extract_any_result.return_value().has_value()) {
            EXPECT_EQ(extract_any_result.return_value()->first, this_pid);
            EXPECT_EQ(extract_result.return_value(), std::nullopt);
        } else {
            // else extract_result should be valid and an empty vector
            EXPECT_TRUE(extract_result.return_value().has_value());
            EXPECT_EQ(extract_result.return_value()->size(), 0);
        }
    }
};

TEST_F(CompetingShufflerAsyncTest, extract_any_then_extract) {
    EXPECT_NO_FATAL_FAILURE(run_test([&](auto shuffler, auto this_pid) {
        return coro::sync_wait(
            coro::when_all(
                shuffler->extract_any_async(), shuffler->extract_async(this_pid)
            )
        );
    }));
}

TEST_F(CompetingShufflerAsyncTest, extract_then_extract_any) {
    EXPECT_NO_FATAL_FAILURE(run_test([&](auto shuffler, auto this_pid) {
        auto [extract_result, extract_any_result] = coro::sync_wait(
            coro::when_all(
                shuffler->extract_async(this_pid), shuffler->extract_any_async()
            )
        );
        // rotate the results to match the order of the coroutines
        return std::make_tuple(std::move(extract_any_result), std::move(extract_result));
    }));
}
