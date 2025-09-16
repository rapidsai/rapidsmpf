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

TEST_P(StreamingShuffler, blocking_shuffler) {
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

TEST_P(StreamingShuffler, shuffler_async) {
    EXPECT_NO_FATAL_FAILURE(
        run_test([&](auto ctx, auto ch_in, auto ch_out, std::vector<Node>& nodes) {
            auto shuffler_async = std::make_shared<ShufflerAsync>(
                std::move(ctx), stream, op_id, num_partitions
            );
            nodes.emplace_back(
                node::shuffler_async_insert(shuffler_async, stream, std::move(ch_in))
            );
            nodes.emplace_back(
                node::shuffler_async_extract(std::move(shuffler_async), std::move(ch_out))
            );
        })
    );
}

TEST_P(StreamingShuffler, shuffler_async_extract_by_pid) {
    auto extract_by_pid = [](auto shuffler_async, auto ch_out) -> Node {
        auto& ctx = shuffler_async->ctx();
        co_await ctx->executor()->schedule();

        auto comm = ctx->comm();
        for (shuffler::PartID pid = 0; pid < shuffler_async->total_num_partitions();
             ++pid)
        {
            if (shuffler_async->partition_owner()(comm, pid) != comm->rank()) {
                continue;
            }

            auto chunks = co_await shuffler_async->extract_async(pid);

            co_await ch_out->send(
                std::make_unique<PartitionVectorChunk>(pid, std::move(chunks))
            );
        }

        co_await ch_out->drain(ctx->executor());
    };

    EXPECT_NO_FATAL_FAILURE(
        run_test([&](auto ctx, auto ch_in, auto ch_out, std::vector<Node>& nodes) {
            auto shuffler_async = std::make_shared<ShufflerAsync>(
                std::move(ctx), stream, op_id, num_partitions
            );
            nodes.emplace_back(
                node::shuffler_async_insert(shuffler_async, stream, std::move(ch_in))
            );
            nodes.emplace_back(
                extract_by_pid(std::move(shuffler_async), std::move(ch_out))
            );
        })
    );
}

TEST_P(StreamingShuffler, multiple_consumers) {
    auto gen_partitions_task = [](std::shared_ptr<Context> ctx,
                                  rmm::cuda_stream_view stream,
                                  size_t n_inserts,
                                  uint32_t num_partitions,
                                  std::shared_ptr<Channel> ch_out) -> Node {
        co_await ctx->executor()->schedule();

        auto br = ctx->br();
        for (size_t i = 0; i < n_inserts; ++i) {
            std::unordered_map<shuffler::PartID, PackedData> data;
            for (shuffler::PartID pid = 0; pid < num_partitions; ++pid) {
                auto [res, ob] = br->reserve(MemoryType::DEVICE, 100, true);
                data.emplace(
                    pid,
                    PackedData(
                        std::make_unique<std::vector<std::uint8_t>>(100),
                        br->allocate(stream, std::move(res))
                    )
                );
            }
            co_await ch_out->send(
                std::make_unique<PartitionMapChunk>(i, std::move(data))
            );
        }
        co_await ch_out->drain(ctx->executor());
    };

    auto consume_partitions_task = [](std::shared_ptr<Context> ctx,
                                      std::vector<std::shared_ptr<Channel>>& chs_in,
                                      size_t exp_local_partitions) -> Node {
        co_await ctx->executor()->schedule();

        size_t n_partitions_received = 0;
        while (!chs_in.empty()) {
            for (auto& ch_in : chs_in) {
                auto msg = co_await ch_in->receive();
                if (msg.empty()) {  // channel is finished
                    ch_in = nullptr;
                } else {
                    auto partition_vec = msg.template release<PartitionVectorChunk>();
                    n_partitions_received += partition_vec.data.size();
                }
            }

            std::erase(chs_in, nullptr);  // remove finished channels
        }
        EXPECT_EQ(n_partitions_received, exp_local_partitions);
        co_return;
    };

    size_t n_inserts = 10;
    uint32_t n_partitions = 20;
    size_t n_consumers = 5;
    auto local_pids = shuffler::Shuffler::local_partitions(
        ctx->comm(), n_partitions, shuffler::Shuffler::round_robin
    );

    /*
     * Data flow graph illustration:
     *
     *                              +-------------------+
     *                              |   gen_partitions  |
     *                              +---------+---------+
     *                                        |
     *                                        | shuffler_in
     *                                        v
     *                              +-------------------+
     *                              |    insert_node    |
     *                              +---------+---------+

     *                  +---------------+ +---------------+ +---------------+
     *                  |extract_node   | |extract_node   | |extract_node   |
     *                  |    [0]        | |    [1]        | |   [n-1]       |
     *                  +-------+-------+ +-------+-------+ +-------+-------+
     *                          |                 |                 |
     *                          | shuffler_out[0] | shuffler_out[1] | shuffler_out[n-1]
     *                          v                 v                 v
     *                          |                 |                 |
     *                          +---------+-------+---------+-------+
     *                                    |                 |
     *                                    v                 v
     *                                 +---------------------+
     *                                 |  consume_partitions |
     *                                 +---------------------+
     *
     * This graph tests multiple consumers extracting partitions concurrently
     * from a single ShufflerAsync instance.
     */

    auto shuffler_in = std::make_shared<Channel>();
    std::vector<std::shared_ptr<Channel>> shuffler_out(n_consumers);
    for (size_t i = 0; i < n_consumers; ++i) {
        shuffler_out[i] = std::make_shared<Channel>();
    }

    auto shuffler_async =
        std::make_shared<ShufflerAsync>(ctx, stream, op_id, n_partitions);

    std::vector<Node> nodes;

    nodes.emplace_back(
        gen_partitions_task(ctx, stream, n_inserts, n_partitions, shuffler_in)
    );

    nodes.emplace_back(
        node::shuffler_async_insert(shuffler_async, stream, std::move(shuffler_in))
    );

    for (size_t i = 0; i < n_consumers; ++i) {
        nodes.emplace_back(node::shuffler_async_extract(shuffler_async, shuffler_out[i]));
    }

    nodes.emplace_back(
        consume_partitions_task(ctx, shuffler_out, n_inserts * local_pids.size())
    );

    run_streaming_pipeline(std::move(nodes));
}
