/**

 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cuda/std/chrono>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/context.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "bloom_filter.hpp"
#include "concatenate.hpp"
#include "groupby.hpp"
#include "join.hpp"
#include "parquet_writer.hpp"
#include "sort.hpp"
#include "utils.hpp"

namespace {

std::vector<rapidsmpf::ndsh::groupby_request> chunkwise_groupby_requests() {
    auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
    std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
    // count(*)
    aggs.emplace_back([]() {
        return cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::INCLUDE
        );
    });
    requests.emplace_back(0, std::move(aggs));
    return requests;
}

std::vector<rapidsmpf::ndsh::groupby_request> final_groupby_requests() {
    auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
    std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
    // sum of partial counts
    aggs.emplace_back([]() {
        return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    });
    requests.emplace_back(1, std::move(aggs));  // column 1 is order_count
    return requests;
}

/* Select the columns after the join

Input table:

- o_orderkey
- o_orderpriority

*/
rapidsmpf::streaming::Node select_columns(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::vector<cudf::size_type> indices
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();

    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            ctx->comm()->logger().debug("Select columns: no more input");
            break;
        }
        auto chunk =
            co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
        auto chunk_stream = chunk.stream();
        auto sequence_number = msg.sequence_number();
        auto table = chunk.table_view();

        auto result_table = std::make_unique<cudf::table>(
            chunk.table_view().select(indices), chunk_stream, ctx->br()->device_mr()
        );

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence_number,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(result_table), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node read_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "lineitem")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({
                           "l_commitdate",  // used in filter
                           "l_receiptdate",  // used in filter
                           "l_orderkey",  // used in join
                       })
                       .build();

    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_orders(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory,
    bool use_date32
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "orders")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({
                           "o_orderkey",  // used in join
                           "o_orderpriority",  // used in group by
                       })
                       .build();

    auto stream = ctx->br()->stream_pool().get_stream();
    // 1993-07-01 <= o_orderdate < 1993-10-01
    constexpr auto start_date = cuda::std::chrono::year_month_day(
        cuda::std::chrono::year(1993),
        cuda::std::chrono::month(7),
        cuda::std::chrono::day(1)
    );
    constexpr auto end_date = cuda::std::chrono::year_month_day(
        cuda::std::chrono::year(1993),
        cuda::std::chrono::month(10),
        cuda::std::chrono::day(1)
    );
    auto filter = use_date32
                      ? rapidsmpf::ndsh::make_date_range_filter<cudf::timestamp_D>(
                            stream, start_date, end_date, "o_orderdate"
                        )
                      : rapidsmpf::ndsh::make_date_range_filter<cudf::timestamp_ms>(
                            stream, start_date, end_date, "o_orderdate"
                        );

    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter)
    );
}

rapidsmpf::streaming::Node filter_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto mr = ctx->br()->device_mr();

    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk =
            co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();

        auto l_commitdate = table.column(0);
        auto l_receiptdate = table.column(1);
        auto mask = cudf::binary_operation(
            l_commitdate,
            l_receiptdate,
            cudf::binary_operator::LESS,
            cudf::data_type(cudf::type_id::BOOL8),
            chunk_stream,
            mr
        );
        auto filtered_table =
            cudf::apply_boolean_mask(table.select({2}), mask->view(), chunk_stream, mr);
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                msg.sequence_number(),
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(filtered_table), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

[[maybe_unused]]
rapidsmpf::streaming::Node fanout_bounded(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch1_out,
    std::vector<cudf::size_type> ch1_cols,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch2_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch1_out, ch2_out};
    co_await ctx->executor()->schedule();

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk =
            co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(
                ctx
            );  // Here, we know that copying ch1_cols (a single col) is better than
                // copying
        // ch2_cols (the whole table)
        std::vector<coro::task<bool>> tasks;
        if (!ch1_out->is_shutdown()) {
            auto msg1 = rapidsmpf::streaming::to_message(
                msg.sequence_number(),
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(
                        chunk.table_view().select(ch1_cols),
                        chunk.stream(),
                        ctx->br()->device_mr()
                    ),
                    chunk.stream()
                )
            );
            tasks.push_back(ch1_out->send(std::move(msg1)));
        }
        if (!ch2_out->is_shutdown()) {
            // TODO: We know here that ch2 wants the whole table.
            tasks.push_back(ch2_out->send(
                rapidsmpf::streaming::to_message(
                    msg.sequence_number(),
                    std::make_unique<rapidsmpf::streaming::TableChunk>(std::move(chunk))
                )
            ));
        }
        if (!std::ranges::any_of(
                rapidsmpf::streaming::coro_results(
                    co_await coro::when_all(std::move(tasks))
                ),
                std::identity{}
            ))
        {
            ctx->comm()->logger().print("Breaking after ", msg.sequence_number());
            break;
        };
    }

    rapidsmpf::streaming::coro_results(
        co_await coro::when_all(
            ch1_out->drain(ctx->executor()), ch2_out->drain(ctx->executor())
        )
    );
}

}  // namespace

/**
 * @brief Run a derived version of TPCH-query 4.
 *
 * The SQL form of the query is:
 * @code{.sql}
 *
 * SELECT
 *     o_orderpriority,
 *     count(*) as order_count
 * FROM
 *     orders
 * where
 *     o_orderdate >= TIMESTAMP '1993-07-01'
 *     and o_orderdate < TIMESTAMP '1993-07-01' + INTERVAL '3' MONTH
 *     and EXISTS (
 *         SELECT
 *             *
 *         FROM
 *             lineitem
 *         WHERE
 *             l_orderkey = o_orderkey
 *             and l_commitdate < l_receiptdate
 *     )
 * GROUP BY
 *     o_orderpriority
 * ORDER BY
 *     o_orderpriority
 * @endcode{}
 *
 * The "exists" clause is translated into a left-semi join in libcudf.
 */
int main(int argc, char** argv) {
    cudaFree(nullptr);

    rapidsmpf::ndsh::FinalizeMPI finalize{};
    cudaFree(nullptr);
    // work around https://github.com/rapidsai/cudf/issues/20849
    cudf::initialize();
    auto mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_wrapper = rapidsmpf::RmmResourceAdaptor(&mr);
    auto arguments = rapidsmpf::ndsh::parse_arguments(argc, argv);
    auto ctx = rapidsmpf::ndsh::create_context(arguments, &stats_wrapper);
    std::string output_path = arguments.output_file;
    std::vector<double> timings;

    // Detect date column types from parquet metadata before timed section
    auto const orders_types =
        rapidsmpf::ndsh::detail::get_column_types(arguments.input_directory, "orders");
    bool const orders_use_date32 =
        orders_types.at("o_orderdate").id() == cudf::type_id::TIMESTAMP_DAYS;

    int l2size;
    int device;
    RAPIDSMPF_CUDA_TRY(cudaGetDevice(&device));
    RAPIDSMPF_CUDA_TRY(cudaDeviceGetAttribute(&l2size, cudaDevAttrL2CacheSize, device));
    auto const num_filter_blocks = rapidsmpf::ndsh::BloomFilter::fitting_num_blocks(
        static_cast<std::size_t>(l2size)
    );

    for (int i = 0; i < arguments.num_iterations; i++) {
        rapidsmpf::OpID op_id{0};
        std::vector<rapidsmpf::streaming::Node> nodes;
        auto start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q4 pipeline");
            // Convention for channel names: express the *output*.
            /* Lineitem Table */
            // [l_commitdate, l_receiptdate, l_orderkey]
            auto lineitem = ctx->create_channel();
            // [l_orderkey]
            auto filtered_lineitem = ctx->create_channel();
            // [l_orderkey]
            auto filtered_lineitem_shuffled = ctx->create_channel();

            /* Orders Table */
            // [o_orderkey, o_orderpriority]
            auto order = ctx->create_channel();

            // [o_orderkey, o_orderpriority]
            // Ideally this would *just* be o_orderpriority, pushing the projection
            // into the join node / dropping the join key.
            auto orders_x_lineitem = ctx->create_channel();

            // [o_orderpriority]
            auto projected_columns = ctx->create_channel();
            // [o_orderpriority, order_count]
            auto grouped_chunkwise = ctx->create_channel();

            nodes.push_back(read_lineitem(
                ctx, lineitem, 4, arguments.num_rows_per_chunk, arguments.input_directory
            ));
            nodes.push_back(
                filter_lineitem(ctx, lineitem, filtered_lineitem)
            );  // l_orderkey
            nodes.push_back(read_orders(
                ctx,
                order,
                4,
                arguments.num_rows_per_chunk,
                arguments.input_directory,
                orders_use_date32
            ));

            // Fanout filtered orders: one for bloom filter, one for join
            auto bloom_filter_input = ctx->create_channel();
            auto orders_for_join = ctx->create_channel();
            nodes.push_back(
                fanout_bounded(ctx, order, bloom_filter_input, {0}, orders_for_join)
            );

            // Build bloom filter from filtered orders' o_orderkey
            auto bloom_filter_output = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::build_bloom_filter(
                    ctx,
                    bloom_filter_input,
                    bloom_filter_output,
                    static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                    cudf::DEFAULT_HASH_SEED,
                    num_filter_blocks
                )
            );

            // Apply bloom filter to filtered lineitem before shuffling
            auto bloom_filtered_lineitem = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::apply_bloom_filter(
                    ctx,
                    bloom_filter_output,
                    filtered_lineitem,
                    bloom_filtered_lineitem,
                    {0}
                )
            );

            // We unconditionally shuffle the filtered lineitem table. This is
            // necessary to correctly handle duplicates in the left-semi join.
            // Failing to shuffle (hash partition) the right table on the join
            // key could allow a record to match multiple times from the
            // multiple partitions of the right table.

            // TODO: configurable num_partitions
            std::uint32_t num_partitions = 16;
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    bloom_filtered_lineitem,
                    filtered_lineitem_shuffled,
                    {0},
                    num_partitions,
                    static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                )
            );

            if (arguments.use_shuffle_join) {
                auto filtered_order_shuffled = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::shuffle(
                        ctx,
                        orders_for_join,
                        filtered_order_shuffled,
                        {0},
                        num_partitions,
                        static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                    )
                );

                nodes.push_back(
                    rapidsmpf::ndsh::left_semi_join_shuffle(
                        ctx,
                        filtered_order_shuffled,
                        filtered_lineitem_shuffled,
                        orders_x_lineitem,
                        {0},
                        {0}
                    )
                );
            } else {
                nodes.push_back(
                    rapidsmpf::ndsh::left_semi_join_broadcast_left(
                        ctx,
                        orders_for_join,
                        filtered_lineitem_shuffled,
                        orders_x_lineitem,
                        {0},
                        {0},
                        static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                        rapidsmpf::ndsh::KeepKeys::YES
                    )
                );
            }

            nodes.push_back(
                select_columns(ctx, orders_x_lineitem, projected_columns, {1})
            );

            nodes.push_back(
                rapidsmpf::ndsh::chunkwise_group_by(
                    ctx,
                    projected_columns,
                    grouped_chunkwise,
                    {0},
                    chunkwise_groupby_requests(),
                    cudf::null_policy::INCLUDE
                )
            );
            auto final_groupby_input = ctx->create_channel();
            if (ctx->comm()->nranks() > 1) {
                nodes.push_back(
                    rapidsmpf::ndsh::broadcast(
                        ctx,
                        grouped_chunkwise,
                        final_groupby_input,
                        static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                        rapidsmpf::streaming::AllGather::Ordered::NO
                    )
                );
            } else {
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx, grouped_chunkwise, final_groupby_input
                    )
                );
            }
            if (ctx->comm()->rank() == 0) {
                auto final_groupby_output = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::chunkwise_group_by(
                        ctx,
                        final_groupby_input,
                        final_groupby_output,
                        {0},
                        final_groupby_requests(),
                        cudf::null_policy::INCLUDE
                    )
                );
                auto sorted_output = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::chunkwise_sort_by(
                        ctx,
                        final_groupby_output,
                        sorted_output,
                        {0},
                        {0, 1},
                        {cudf::order::ASCENDING},
                        {cudf::null_order::BEFORE}
                    )
                );
                nodes.push_back(
                    rapidsmpf::ndsh::write_parquet(
                        ctx,
                        sorted_output,
                        cudf::io::sink_info(output_path),
                        {"o_orderpriority", "order_count"}
                    )
                );
            } else {
                nodes.push_back(rapidsmpf::ndsh::sink_channel(ctx, final_groupby_input));
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> pipeline = end - start;
        start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("Q4 Iteration");
            rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
        }
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> compute = end - start;
        timings.push_back(pipeline.count());
        timings.push_back(compute.count());
        ctx->comm()->logger().print(ctx->statistics()->report());
        ctx->statistics()->clear();
    }

    if (ctx->comm()->rank() == 0) {
        for (int i = 0; i < arguments.num_iterations; i++) {
            ctx->comm()->logger().print(
                "Iteration ",
                i,
                " pipeline construction time [s]: ",
                timings[size_t(2 * i)]
            );
            ctx->comm()->logger().print(
                "Iteration ", i, " compute time [s]: ", timings[size_t(2 * i + 1)]
            );
        }
    }
    return 0;
}
