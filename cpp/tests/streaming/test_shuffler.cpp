/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ranges>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "../utils.hpp"
#include "base_streaming_fixture.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::streaming;
namespace node = rapidsmpf::streaming::node;

class StreamingShuffler : public BaseStreamingFixture,
                          public ::testing::WithParamInterface<int> {
  public:
    const unsigned int num_partitions = 10;
    const unsigned int num_rows = 1000;
    const unsigned int num_chunks = 5;
    const unsigned int chunk_size = num_rows / num_chunks;
    const std::int64_t seed = 42;
    const cudf::hash_id hash_function = cudf::hash_id::HASH_MURMUR3;
    const OpID op_id = 0;

    // override the base SetUp
    void SetUp() override {
        BaseStreamingFixture::SetUpWithThreads(GetParam());
        GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
        BaseStreamingFixture::TearDown();
    }

    void run_test(auto make_shuffler_node_fn) {
        // Create the full input table and slice it into chunks.
        cudf::table full_input_table = random_table_with_index(seed, num_rows, 0, 10);
        std::vector<Message> input_chunks;
        for (unsigned int i = 0; i < num_chunks; ++i) {
            input_chunks.emplace_back(
                std::make_unique<TableChunk>(
                    i,
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
            );
        }

        // Create and run the streaming pipeline.
        std::vector<Message> output_chunks;
        {
            std::vector<Node> nodes;
            auto ch1 = std::make_shared<Channel>();
            nodes.push_back(node::push_to_channel(ctx, ch1, std::move(input_chunks)));

            auto ch2 = std::make_shared<Channel>();
            nodes.push_back(
                node::partition_and_pack(
                    ctx, ch1, ch2, {1}, num_partitions, hash_function, seed
                )
            );

            auto ch3 = std::make_shared<Channel>();
            nodes.emplace_back(make_shuffler_node_fn(ch2, ch3));

            auto ch4 = std::make_shared<Channel>();
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
            offsets.push_back(table->num_rows());
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
        return node::shuffler(
            ctx, std::move(ch_in), std::move(ch_out), op_id, num_partitions
        );
    }));
}

namespace {

// emulate shuffler node with callbacks
Node shuffler_nb(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    int n_consumers
) {
    struct ShufflerContext {
        std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler{};

        // queue that holds the partition ids that are ready to be extracted. Progress
        // thread will push the partition ids to the queue. The extract task will pop the
        // partition ids from the queue and extract the chunks from the shuffler.
        coro::queue<rapidsmpf::shuffler::PartID> ready_pids{};

        coro::task<void> push_to_queue(rapidsmpf::shuffler::PartID pid) {
            auto result = co_await ready_pids.push(pid);
            RAPIDSMPF_EXPECTS(
                result == coro::queue_produce_result::produced,
                "failed to push partition id to ready_pids"
            );
        }
    };

    // make a shared_ptr to the shuffler_ctx so that it can be passed into multiple
    // coroutines
    auto shuffler_ctx = std::make_shared<ShufflerContext>();
    shuffler_ctx->shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
        ctx->comm(),
        ctx->progress_thread(),
        op_id,
        total_num_partitions,
        ctx->br(),
        [ctx_ptr = ctx.get(),
         shuffler_ctx_ptr = shuffler_ctx.get()](rapidsmpf::shuffler::PartID pid) {
            // detached task to push the partition id to the queue
            RAPIDSMPF_EXPECTS(
                ctx_ptr->executor()->spawn(shuffler_ctx_ptr->push_to_queue(pid)),
                "failed to spawn task to push partition id to ready_pids"
            );
        },
        ctx->statistics(),
        shuffler::Shuffler::round_robin
    );

    // insert task: insert the partition map chunks into the shuffler
    auto insert_task =
        [](auto shuffler_ctx, auto ctx, auto total_num_partitions, auto ch_in) -> Node {
        ShutdownAtExit c{ch_in};
        co_await ctx->executor()->schedule();

        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            auto partition_map = msg.template release<PartitionMapChunk>();
            shuffler_ctx->shuffler->insert(std::move(partition_map.data));
        }

        // Tell the shuffler that we have no more input data.
        std::vector<rapidsmpf::shuffler::PartID> finished(total_num_partitions);
        std::iota(finished.begin(), finished.end(), 0);
        shuffler_ctx->shuffler->insert_finished(std::move(finished));
        co_return;
    };

    // extract task: extract the packed chunks from the shuffler and send them to the
    // output channel
    auto extract_task =
        [](auto shuffler_ctx, auto ctx, auto ch_out, auto& latch) -> Node {
        co_await ctx->executor()->schedule();

        while (!shuffler_ctx->shuffler->finished()) {
            auto pid = co_await shuffler_ctx->ready_pids.pop();
            if (!pid) {
                break;  // queue is shutdown, so exit the loop
            }

            auto packed_chunks = shuffler_ctx->shuffler->extract(*pid);

            co_await ch_out->send(
                std::make_unique<PartitionVectorChunk>(*pid, std::move(packed_chunks))
            );

            if (shuffler_ctx->shuffler->finished()) {
                // if the shuffler is finished, shutdown & drain the ready_pids queue
                co_await shuffler_ctx->ready_pids.shutdown_drain(ctx->executor());
            }
        }

        latch.count_down();  // this task is finished, so count down the latch
    };

    // shutdown task: shutdown the shuffler after all extract tasks have finished
    auto shutdown_task =
        [](auto shuffler_ctx, auto ctx, auto ch_out, auto& latch) -> Node {
        ShutdownAtExit c{ch_out};
        co_await ctx->executor()->schedule();

        co_await latch;  // wait for all extract tasks to finish before clean up
        co_await ch_out->drain(ctx->executor());

        shuffler_ctx->shuffler->shutdown();
    };

    std::vector<Node> nodes;
    coro::latch latch(n_consumers);

    nodes.emplace_back(
        insert_task(shuffler_ctx, ctx, total_num_partitions, std::move(ch_in))
    );
    for (int i = 0; i < n_consumers; ++i) {
        nodes.emplace_back(extract_task(shuffler_ctx, ctx.get(), ch_out, latch));
    }
    nodes.emplace_back(
        shutdown_task(std::move(shuffler_ctx), ctx.get(), std::move(ch_out), latch)
    );

    co_await coro::when_all(std::move(nodes));
}

}  // namespace

TEST_P(StreamingShuffler, callbacks_1_consumer) {
    EXPECT_NO_FATAL_FAILURE(run_test([&](auto ch_in, auto ch_out) -> Node {
        return shuffler_nb(
            ctx, std::move(ch_in), std::move(ch_out), op_id, num_partitions, 1
        );
    }));
}

TEST_P(StreamingShuffler, callbacks_2_consumer) {
    EXPECT_NO_FATAL_FAILURE(run_test([&](auto ch_in, auto ch_out) -> Node {
        return shuffler_nb(
            ctx, std::move(ch_in), std::move(ch_out), op_id, num_partitions, 2
        );
    }));
}

TEST_P(StreamingShuffler, callbacks_4_consumer) {
    EXPECT_NO_FATAL_FAILURE(run_test([&](auto ch_in, auto ch_out) -> Node {
        return shuffler_nb(
            ctx, std::move(ch_in), std::move(ch_out), op_id, num_partitions, 4
        );
    }));
}

class ShufflerAsyncTest
    : public BaseStreamingFixture,
      public ::testing::WithParamInterface<std::tuple<int, size_t, uint32_t, int>> {
  protected:
    int n_threads;
    size_t n_inserts;
    uint32_t n_partitions;
    int n_consumers;

    std::unique_ptr<ShufflerAsync> shuffler;

    static constexpr OpID op_id = 0;
    static constexpr size_t n_elements = 100;

    void SetUp() override {
        std::tie(n_threads, n_inserts, n_partitions, n_consumers) = GetParam();
        BaseStreamingFixture::SetUpWithThreads(n_threads);
        GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers

        shuffler = std::make_unique<ShufflerAsync>(ctx, op_id, n_partitions);
    }

    void TearDown() override {
        shuffler.reset();
        BaseStreamingFixture::TearDown();
        GlobalEnvironment->barrier();
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
    // extract data (executed by thread pool)
    auto extract_task = [](int tid,
                           auto* shuffler,
                           auto* ctx,
                           coro::mutex& mtx,
                           std::vector<shuffler::PartID>& finished_pids,
                           size_t& n_chunks_received) -> coro::task<void> {
        co_await ctx->executor()->schedule();
        ctx->comm()->logger().debug(tid, " extract task started");

        while (!shuffler->finished()) {
            auto result = co_await shuffler->extract_any_async();
            if (!result.has_value()) {
                break;
            }

            auto lock = co_await mtx.scoped_lock();
            auto& [pid, chunks] = *result;
            n_chunks_received += chunks.size();
            finished_pids.push_back(pid);
        }
        ctx->comm()->logger().debug(tid, " extract task finished");
    };

    // insert data (executed by main thread)
    for (size_t i = 0; i < n_inserts; ++i) {
        std::unordered_map<shuffler::PartID, PackedData> data;
        data.reserve(n_partitions);
        for (shuffler::PartID pid = 0; pid < n_partitions; ++pid) {
            data.emplace(pid, generate_packed_data(n_elements, 0, stream, *br));
        }
        shuffler->insert(std::move(data));
    }

    // insert finished (executed by main thread)
    shuffler->insert_finished(iota_vector<shuffler::PartID>(n_partitions));

    coro::mutex mtx;
    std::vector<shuffler::PartID> finished_pids;
    size_t n_chunks_received = 0;
    std::vector<Node> extract_tasks;
    for (int i = 0; i < n_consumers; ++i) {
        extract_tasks.emplace_back(extract_task(
            i, shuffler.get(), ctx.get(), mtx, finished_pids, n_chunks_received
        ));
    }

    // wait for the extract task to finish (executed by thread pool, waited by main
    // thread)
    run_streaming_pipeline(std::move(extract_tasks));

    auto local_pids = shuffler::Shuffler::local_partitions(
        ctx->comm(), n_partitions, shuffler::Shuffler::round_robin
    );
    EXPECT_EQ(n_inserts * local_pids.size() * ctx->comm()->nranks(), n_chunks_received);

    std::ranges::sort(finished_pids);
    EXPECT_EQ(local_pids, finished_pids);

    GlobalEnvironment->barrier();  // wait for all ranks to finish
}

TEST_F(BaseStreamingFixture, extract_any_before_extract) {
    GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
    static constexpr OpID op_id = 0;
    static constexpr size_t n_partitions = 10;
    auto shuffler = std::make_unique<ShufflerAsync>(ctx, op_id, n_partitions);

    // all empty partitions
    shuffler->insert_finished(iota_vector<shuffler::PartID>(n_partitions));

    auto local_pids = shuffler::Shuffler::local_partitions(
        ctx->comm(), n_partitions, shuffler::Shuffler::round_robin
    );

    size_t parts_extracted = 0;
    while (true) {  // extract all partitions
        auto res = coro::sync_wait(shuffler->extract_any_async());
        if (!res.has_value()) {
            break;
        }
        parts_extracted++;
    }
    EXPECT_EQ(local_pids.size(), parts_extracted);

    // now extract should throw
    for (auto pid : local_pids) {
        EXPECT_THROW(coro::sync_wait(shuffler->extract_async(pid)), std::out_of_range);
    }
    shuffler.reset();
    GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
}

class CompetingShufflerAsyncTest : public BaseStreamingFixture {
  protected:
    // produce_results_fn is a function that produces the results of the extract_any_async
    // and extract_async coroutines.
    void run_test(auto produce_results_fn) {
        GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
        static constexpr OpID op_id = 0;
        shuffler::PartID const n_partitions = ctx->comm()->nranks();
        shuffler::PartID const this_pid = ctx->comm()->rank();

        auto shuffler = std::make_unique<ShufflerAsync>(ctx, op_id, n_partitions);

        shuffler->insert_finished(iota_vector<shuffler::PartID>(n_partitions));

        auto [extract_any_result, extract_result] =
            produce_results_fn(shuffler.get(), this_pid);

        // if extract_any_result is valid, then extract_result should throw
        if (extract_any_result.return_value().has_value()) {
            EXPECT_EQ(extract_any_result.return_value()->first, this_pid);
            EXPECT_THROW(extract_result.return_value(), std::out_of_range);
        } else {
            // else extract_result should be valid and an empty vector
            EXPECT_EQ(extract_result.return_value().size(), 0);
        }
        shuffler.reset();
        GlobalEnvironment->barrier();  // prevent accidental mixup between shufflers
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
