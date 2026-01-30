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
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/context.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
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

/* Sort the grouped orders table by o_orderpriority.

Input table:
    - o_orderpriority
    - order_count

Output table:
    - o_orderpriority (sorted ascending)
    - order_count
*/
rapidsmpf::streaming::Node sort_by(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    }
    auto chunk =
        co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
    auto stream = chunk.stream();
    auto mr = ctx->br()->device_mr();
    auto table = chunk.table_view();

    auto sorted_table = cudf::sort_by_key(
        table,
        table.select({0}),
        {cudf::order::ASCENDING},
        {cudf::null_order::BEFORE},
        stream,
        mr
    );

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            msg.sequence_number(),
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(sorted_table), stream
            )
        )
    );
    co_await ch_out->drain(ctx->executor());
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

    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            ctx->comm()->logger().debug("Select columns: no more input");
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk =
            co_await msg.release<rapidsmpf::streaming::TableChunk>().make_available(ctx);
        auto chunk_stream = chunk.stream();
        auto sequence_number = msg.sequence_number();
        auto table = chunk.table_view();

        std::vector<std::unique_ptr<cudf::column>> result;
        result.reserve(indices.size());
        for (auto idx : indices) {
            result.push_back(
                std::make_unique<cudf::column>(
                    table.column(idx), chunk_stream, ctx->br()->device_mr()
                )
            );
        }

        auto result_table = std::make_unique<cudf::table>(std::move(result));

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
    std::string const& input_directory
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

    // Build the filter expression 1993-07-01 <= o_orderdate < 1993-10-01
    cudf::timestamp_ms ts1{
        cuda::std::chrono::duration_cast<cuda::std::chrono::milliseconds>(
            cuda::std::chrono::sys_days(
                cuda::std::chrono::year_month_day(
                    cuda::std::chrono::year(1993),
                    cuda::std::chrono::month(7),
                    cuda::std::chrono::day(1)
                )
            )
                .time_since_epoch()
        )
    };
    cudf::timestamp_ms ts2{
        cuda::std::chrono::duration_cast<cuda::std::chrono::milliseconds>(
            cuda::std::chrono::sys_days(
                cuda::std::chrono::year_month_day(
                    cuda::std::chrono::year(1993),
                    cuda::std::chrono::month(10),
                    cuda::std::chrono::day(1)
                )
            )
                .time_since_epoch()
        )
    };

    /* This vector will have the references for the expression `a < column < b` as

    0: column_reference to o_orderdate
    1: scalar<ts1>
    2: scalar<ts2>
    3: literal<ts1>
    4: literal<ts2>
    5: operation GE
    6: operation LT
    7: operation AND
    */

    auto owner = new std::vector<std::any>;
    auto filter_stream = ctx->br()->stream_pool().get_stream();
    // 0
    owner->push_back(
        std::make_shared<cudf::ast::column_name_reference>(
            "o_orderdate"
        )  // position in the table
    );


    // 1, 2: Scalars
    owner->push_back(
        std::make_shared<cudf::timestamp_scalar<cudf::timestamp_ms>>(
            ts1, true, filter_stream
        )
    );
    owner->push_back(
        std::make_shared<cudf::timestamp_scalar<cudf::timestamp_ms>>(
            ts2, true, filter_stream
        )
    );

    // 3, 4: Literals
    owner->push_back(
        std::make_shared<cudf::ast::literal>(
            *std::any_cast<std::shared_ptr<cudf::timestamp_scalar<cudf::timestamp_ms>>>(
                owner->at(1)
            )
        )
    );
    owner->push_back(
        std::make_shared<cudf::ast::literal>(
            *std::any_cast<std::shared_ptr<cudf::timestamp_scalar<cudf::timestamp_ms>>>(
                owner->at(2)
            )
        )
    );

    // 5: (GE, column, literal<var1>)
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            cudf::ast::ast_operator::GREATER_EQUAL,
            *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(
                owner->at(0)
            ),
            *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(3))
        )
    );

    // 6 (LT, column, literal<var2>)
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            cudf::ast::ast_operator::LESS,
            *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(
                owner->at(0)
            ),
            *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(4))
        )
    );

    // 7 (AND, GE, LT)
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            cudf::ast::ast_operator::LOGICAL_AND,
            *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->at(5)),
            *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->at(6))
        )
    );

    auto filter = std::make_unique<rapidsmpf::streaming::Filter>(
        filter_stream,
        *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
        rapidsmpf::OwningWrapper(static_cast<void*>(owner), [](void* p) {
            delete static_cast<std::vector<std::any>*>(p);
        })
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
    auto mr = ctx->br()->device_mr();
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
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
                ctx, order, 4, arguments.num_rows_per_chunk, arguments.input_directory
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
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
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
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
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
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
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
                nodes.push_back(sort_by(ctx, final_groupby_output, sorted_output));
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
