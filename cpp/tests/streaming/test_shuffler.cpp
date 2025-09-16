/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/leaf_node.hpp>
#include <rapidsmpf/streaming/cudf/partition.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>
#include <rapidsmpf/streaming/cudf/utils.hpp>

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
        int num_streaming_threads = GetParam();
        rapidsmpf::config::Options options{
            rapidsmpf::config::get_environment_variables()
        };
        options.insert_if_absent(
            "num_streaming_threads", std::to_string(num_streaming_threads)
        );
        stream = cudf::get_default_stream();
        br = std::make_unique<rapidsmpf::BufferResource>(mr_cuda);
        ctx = std::make_shared<rapidsmpf::streaming::Context>(
            std::move(options), std::make_shared<rapidsmpf::Single>(options), br.get()
        );
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
            make_shuffler_node_fn(ctx, ch2, ch3, nodes);

            auto ch4 = std::make_shared<Channel>();
            nodes.push_back(node::unpack_and_concat(ctx, ch3, ch4));

            nodes.push_back(node::pull_from_channel(ctx, ch4, output_chunks));

            run_streaming_pipeline(std::move(nodes));
        }

        // Concat all output chunks to a single table.
        std::vector<cudf::table_view> output_chunks_as_views;
        for (auto& chunk : output_chunks) {
            output_chunks_as_views.push_back(chunk.get<TableChunk>().table_view());
        }
        auto result_table = cudf::concatenate(output_chunks_as_views);

        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(result_table->view()), sort_table(full_input_table.view())
        );
    }
};

INSTANTIATE_TEST_SUITE_P(
    StreamingShuffler,
    StreamingShuffler,
    ::testing::Values(1, 2, 4),
    [](const testing::TestParamInfo<StreamingShuffler::ParamType>& info) {
        return "nthreads_" + std::to_string(info.param);
    }
);

TEST_P(StreamingShuffler, basic_shuffler) {
    EXPECT_NO_FATAL_FAILURE(
        run_test([&](auto ctx, auto ch_in, auto ch_out, std::vector<Node>& nodes) {
            nodes.emplace_back(
                node::shuffler(
                    std::move(ctx),
                    stream,
                    std::move(ch_in),
                    std::move(ch_out),
                    op_id,
                    num_partitions
                )
            );
        })
    );
}

namespace {

// emulate shuffler node with callbacks
std::pair<Node, Node> shuffler_nb(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions
) {
    struct ShufflerContext {
        std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler{};
        coro::queue<rapidsmpf::shuffler::PartID> ready_pids{};
    };

    // make a shared_ptr to the shuffler so that it can be passed into multiple coroutines
    auto shuffler_ctx = std::make_shared<ShufflerContext>();
    shuffler_ctx->shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
        ctx->comm(),
        ctx->progress_thread(),
        op_id,
        total_num_partitions,
        stream,
        ctx->br(),
        [shuffler_ctx](rapidsmpf::shuffler::PartID pid) {
            // synchronously push the partition id to the ready_pids queue
            RAPIDSMPF_EXPECTS(
                coro::sync_wait(shuffler_ctx->ready_pids.push(pid))
                    == coro::queue_produce_result::produced,
                "failed to push partition id to ready_pids"
            );
        },
        ctx->statistics(),
        shuffler::Shuffler::round_robin
    );

    // insert task: insert the partition map chunks into the shuffler
    auto insert_task = [](auto shuffler_ctx,
                          auto ctx,
                          auto total_num_partitions,
                          auto stream,
                          auto ch_in) -> Node {
        ShutdownAtExit c{ch_in};
        co_await ctx->executor()->schedule();
        CudaEvent event;

        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            auto partition_map = msg.template release<PartitionMapChunk>();

            // Make sure that the input chunk's stream is in sync with shuffler's stream.
            utils::sync_streams(stream, partition_map.stream, event);

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
    auto extract_task = [](auto shuffler_ctx, auto ctx, auto ch_out) -> Node {
        ShutdownAtExit c{ch_out};
        co_await ctx->executor()->schedule();

        while (!shuffler_ctx->shuffler->finished() || !shuffler_ctx->ready_pids.empty()) {
            auto expected = co_await shuffler_ctx->ready_pids.pop();
            RAPIDSMPF_EXPECTS(
                expected.has_value(), "failed to pop partition id from ready_pids"
            );

            auto packed_chunks = shuffler_ctx->shuffler->extract(*expected);
            co_await ch_out->send(
                std::make_unique<PartitionVectorChunk>(
                    *expected, std::move(packed_chunks)
                )
            );
        }
        co_await ch_out->drain(ctx->executor());
    };

    return {
        insert_task(shuffler_ctx, ctx, total_num_partitions, stream, std::move(ch_in)),
        extract_task(std::move(shuffler_ctx), std::move(ctx), std::move(ch_out))
    };
}

}  // namespace

TEST_P(StreamingShuffler, callbacks) {
    EXPECT_NO_FATAL_FAILURE(
        run_test([&](auto ctx, auto ch_in, auto ch_out, std::vector<Node>& nodes) {
            auto [insert_node, extract_node] = shuffler_nb(
                std::move(ctx),
                stream,
                std::move(ch_in),
                std::move(ch_out),
                op_id,
                num_partitions
            );
            nodes.emplace_back(std::move(insert_node));
            nodes.emplace_back(std::move(extract_node));
        })
    );
}
