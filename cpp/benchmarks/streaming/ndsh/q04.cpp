/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <mpi.h>

#include <cuda/std/chrono>

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "concatenate.hpp"
#include "join.hpp"
#include "utils.hpp"

/* Query 4 from the TPC-H benchmark.

This performs a left semi join between the orders and lineitem tables,
followed by a grouped count aggregation on a low-cardinality column.

```python
lineitem = pl.scan_parquet("/raid/rapidsmpf/data/tpch/scale-100.0/lineitem.parquet")
orders = pl.scan_parquet("/raid/rapidsmpf/data/tpch/scale-100.0/orders.parquet")

var1 = date(1993, 7, 1)  #  8582
var2 = date(1993, 10, 1)  # 8674

q = (
    # SQL exists translates to semi join in Polars API
    orders.join(
        (lineitem.filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))),
        left_on="o_orderkey",
        right_on="l_orderkey",
        how="semi",
    )
    .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
    .group_by("o_orderpriority")
    .agg(pl.len().alias("order_count"))
    .sort("o_orderpriority")
)
```

Some rough stats at SF-100:

| Scale Factor | Table / Stage     | Row Count   | Percent of prior |
| ------------ | ----------------- | ----------- | ---------------- |
| 100          | lineitem          | 600,037,902 | -                |
| 100          | lineitem-filtered | 379,356,474 | 63%              |
| 100          | orders            | 150,000,000 | -                |
| 100          | orders-filtered   |   5,733,776 | 3.8%             |
| 100          | joined            |   5,257,429 | 91% / 1.4%       |
| 100          | groupby           |           5 | 0.0%             |
| 100          | final             |           5 | 100%             |
| 100          | sorted            |           5 | 100%             |

So the lineitem filter is somewhat selective, the orders filter is very
selective, the join is a bit selective (of orders), and the final groupby
reduces by a lot.

The left-semi join can be performed in one of two ways:

1. Broadcast `orders` to all ranks, shuffle `lineitem`, join per chunk, concat.
2. Shuffle `orders` and `lineitem`, join per chunk, concat

Either way, we *always* shuffle / hash-partition `lineitem` before the join.
We rely on that has partitioning to ensure that the chunkwise left-semi join
is correct (notably, how duplicates are handled).

We don't attempt to reuse the build table (`lineitem`) in the hash partition
for multiple probe table (`orders`) chunks. That would require broadcasting
`lineitem`, which we assume is too large.
*/

namespace {

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
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto sequence_number = msg.sequence_number();
        auto table = chunk.table_view();

        rapidsmpf::ndsh::detail::debug_print_table(ctx, table, "select_columns::input");
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

        rapidsmpf::ndsh::detail::debug_print_table(
            ctx, result_table->view(), "select_columns::output"
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

/*
Read the lineitem table.

Output table:

    - l_commitdate
    - l_receiptdate
    - l_orderkey
*/
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

/*
Read the orders table, including the filter on the o_orderdate column.

Output table:

    - o_orderkey
    - o_orderpriority
*/
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

/* Filter the lineitem table.

Input table:

    - l_commitdate
    - l_receiptdate
    - l_orderkey

Output table:

    - l_orderkey
*/
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
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();

        rapidsmpf::ndsh::detail::debug_print_table(ctx, table, "lineitem");

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
        rapidsmpf::ndsh::detail::debug_print_table(
            ctx, filtered_table->view(), "filtered_lineitem"
        );
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

/* Group the joined orders table by o_orderpriority (chunk).

We're performing a `.group_by(...).count()`, so the chunk-stage
is just a count.

Input table:

    - o_orderkey
    - o_orderpriority

Output table:

    - o_orderpriority
    - order_count
*/
rapidsmpf::streaming::Node chunkwise_groupby_agg(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::vector<cudf::table> partial_results;
    std::uint64_t sequence = 0;
    while (!ch_out->is_shutdown()) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            ctx->comm()->logger().debug("Chunkwise groupby agg: no more input");
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();

        rapidsmpf::ndsh::detail::debug_print_table(
            ctx, table, "chunkwise_groupby_agg::input"
        );

        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(0), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
        // Drop chunk, we don't need it.
        std::ignore = std::move(chunk);
        auto result = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result));
        }

        auto result_table = std::make_unique<cudf::table>(std::move(result));
        rapidsmpf::ndsh::detail::debug_print_table(
            ctx, result_table->view(), "chunkwise_groupby_agg::output"
        );


        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence++,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(result_table), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/* Group the joined orders table by o_orderpriority (final).

We're performing a `.group_by(...).count()`, so the final stage
is just a sum.

Input table:

    - o_orderkey
    - o_orderpriority

Output table:

    - o_orderpriority
    - order_count
*/

rapidsmpf::streaming::Node final_groupby_agg(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    auto next = co_await ch_in->receive();
    ctx->comm()->logger().debug("Final groupby");
    RAPIDSMPF_EXPECTS(next.empty(), "Expecting concatenated input at this point");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();

    rapidsmpf::ndsh::detail::debug_print_table(ctx, table, "final_groupby_agg::input");
    std::unique_ptr<cudf::table> local_result{nullptr};
    if (!table.is_empty()) {
        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        requests.push_back(
            cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
        // Drop chunk, we don't need it.
        std::ignore = std::move(chunk);
        auto result = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result));
        }
        local_result = std::make_unique<cudf::table>(std::move(result));
    }
    if (ctx->comm()->nranks() > 1) {
        // Reduce across ranks...
        // Need a reduce primitive in rapidsmpf, but let's just use an allgather and
        // discard for now.
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};
        if (local_result) {
            auto pack =
                cudf::pack(local_result->view(), chunk_stream, ctx->br()->device_mr());
            gatherer.insert(
                0,
                {rapidsmpf::PackedData(
                    std::move(pack.metadata),
                    ctx->br()->move(std::move(pack.gpu_data), chunk_stream)
                )}
            );
        }
        gatherer.insert_finished();
        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);
        if (ctx->comm()->rank() == 0) {
            auto global_result = rapidsmpf::unpack_and_concat(
                rapidsmpf::unspill_partitions(
                    std::move(packed_data), ctx->br(), true, ctx->statistics()
                ),
                chunk_stream,
                ctx->br(),
                ctx->statistics()
            );
            if (ctx->comm()->rank() == 0) {
                // We will only actually bother to do this on rank zero.
                auto result_view = global_result->view();
                auto grouper = cudf::groupby::groupby(
                    result_view.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
                );
                auto requests = std::vector<cudf::groupby::aggregation_request>();
                std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
                aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
                requests.push_back(
                    cudf::groupby::aggregation_request(
                        result_view.column(1), std::move(aggs)
                    )
                );
                auto [keys, results] =
                    grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
                global_result.reset();
                auto result = keys->release();
                for (auto&& r : results) {
                    std::ranges::move(r.results, std::back_inserter(result));
                }
                co_await ch_out->send(
                    rapidsmpf::streaming::to_message(
                        0,
                        std::make_unique<rapidsmpf::streaming::TableChunk>(
                            std::make_unique<cudf::table>(std::move(result)), chunk_stream
                        )
                    )
                );
            }
        } else {
            std::ignore = std::move(packed_data);
        }
    } else {
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(local_result), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/* Sort the grouped orders table.

Input table:

    - o_orderpriority
    - order_count

Output table:

    - o_orderpriority
    - order_count
*/
rapidsmpf::streaming::Node sort_by(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    ctx->comm()->logger().debug("Final sortby");
    auto msg = co_await ch_in->receive();
    // We know we only have a single chunk from the groupby
    if (msg.empty()) {
        co_return;
    }
    ctx->comm()->logger().debug("Sortby");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto table = chunk.table_view();
    rapidsmpf::ndsh::detail::debug_print_table(ctx, table, "sort_by::input");
    auto result = rapidsmpf::streaming::to_message(
        0,
        std::make_unique<rapidsmpf::streaming::TableChunk>(
            cudf::sort_by_key(
                table,
                table.select({0, 1}),
                {cudf::order::ASCENDING, cudf::order::DESCENDING},
                {cudf::null_order::BEFORE, cudf::null_order::BEFORE},
                chunk.stream(),
                ctx->br()->device_mr()
            ),
            chunk.stream()
        )
    );
    co_await ch_out->send(std::move(result));
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node write_parquet(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::string output_path
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    }
    ctx->comm()->logger().debug("write parquet");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto sink = cudf::io::sink_info(output_path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("o_orderpriority");
    metadata.column_metadata[1].set_name("order_count");
    builder = builder.metadata(metadata);
    auto options = builder.build();
    cudf::io::write_parquet(options, chunk.stream());
    ctx->comm()->logger().print(
        "Wrote chunk with ",
        chunk.table_view().num_rows(),
        " rows and ",
        chunk.table_view().num_columns(),
        " columns to ",
        output_path
    );
}


}  // namespace

struct ProgramOptions {
    int num_streaming_threads{1};
    cudf::size_type num_rows_per_chunk{100'000'000};
    std::optional<double> spill_device_limit{std::nullopt};
    bool use_shuffle_join = false;
    std::string output_file;
    std::string input_directory;
    std::uint32_t num_partitions{16};
};

// TODO: Refactor to common utilities
ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;

    auto print_usage = [&argv]() {
        std::cerr
            << "Usage: " << argv[0] << " [options]\n"
            << "Options:\n"
            << "  --num-streaming-threads <n>  Number of streaming threads (default: 1)\n"
            << "  --num-rows-per-chunk <n>     Number of rows per chunk (default: "
               "100000000)\n"
            << "  --spill-device-limit <n>     Fractional spill device limit (default: "
               "None)\n"
            << "  --use-shuffle-join           Use shuffle join (default: false)\n"
            << "  --output-file <path>         Output file path (required)\n"
            << "  --input-directory <path>     Input directory path (required)\n"
            << "  --num-partitions <n>         Number of partitions (default: 16)\n"
            << "  --help                       Show this help message\n";
    };

    static std::array<option, 9> long_options = {{
        {.name = "num-streaming-threads",
         .has_arg = required_argument,
         .flag = nullptr,
         .val = 1},
        {.name = "num-rows-per-chunk",
         .has_arg = required_argument,
         .flag = nullptr,
         .val = 2},
        {.name = "use-shuffle-join", .has_arg = no_argument, .flag = nullptr, .val = 3},
        {.name = "output-file", .has_arg = required_argument, .flag = nullptr, .val = 4},
        {.name = "input-directory",
         .has_arg = required_argument,
         .flag = nullptr,
         .val = 5},
        {.name = "num-partitions",
         .has_arg = required_argument,
         .flag = nullptr,
         .val = 6},
        {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 6},
        {.name = "spill-device-limit",
         .has_arg = required_argument,
         .flag = nullptr,
         .val = 7},
        {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0},
    }};

    int opt;
    int option_index = 0;

    bool saw_output_file = false;
    bool saw_input_directory = false;

    while ((opt = getopt_long(argc, argv, "", long_options.data(), &option_index)) != -1)
    {
        switch (opt) {
        case 1:
            options.num_streaming_threads = std::atoi(optarg);
            break;
        case 2:
            options.num_rows_per_chunk = std::atoi(optarg);
            break;
        case 3:
            options.use_shuffle_join = true;
            break;
        case 4:
            options.output_file = optarg;
            saw_output_file = true;
            break;
        case 5:
            options.input_directory = optarg;
            saw_input_directory = true;
            break;
        case 6:
            options.num_partitions = static_cast<std::uint32_t>(std::atoi(optarg));
            break;
        case 7:
            print_usage();
            std::exit(0);
        case 8:
            options.spill_device_limit = std::stod(optarg);
            break;
        case '?':
            if (optopt == 0 && optind > 1) {
                std::cerr << "Error: Unknown option '" << argv[optind - 1] << "'\n\n";
            }
            print_usage();
            std::exit(1);
        default:
            print_usage();
            std::exit(1);
        }
    }

    // Check if required options were provided
    if (!saw_output_file || !saw_input_directory) {
        if (!saw_output_file) {
            std::cerr << "Error: --output-file is required\n";
        }
        if (!saw_input_directory) {
            std::cerr << "Error: --input-directory is required\n";
        }
        std::cerr << std::endl;
        print_usage();
        std::exit(1);
    }

    return options;
}

int main(int argc, char** argv) {
    cudaFree(nullptr);
    rapidsmpf::mpi::init(&argc, &argv);
    MPI_Comm mpi_comm;
    RAPIDSMPF_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
    auto cmd_options = parse_options(argc, argv);
    auto limit_size = rmm::percent_of_free_device_memory(
        static_cast<std::size_t>(cmd_options.spill_device_limit.value_or(1) * 100)
    );
    rmm::mr::cuda_async_memory_resource mr{};
    auto stats_mr = rapidsmpf::RmmResourceAdaptor(&mr);
    rmm::device_async_resource_ref mr_ref(stats_mr);
    rmm::mr::set_current_device_resource(&stats_mr);
    rmm::mr::set_current_device_resource_ref(mr_ref);
    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
        memory_available{};
    if (cmd_options.spill_device_limit.has_value()) {
        memory_available[rapidsmpf::MemoryType::DEVICE] = rapidsmpf::LimitAvailableMemory{
            &stats_mr, static_cast<std::int64_t>(limit_size)
        };
    }
    auto br = std::make_shared<rapidsmpf::BufferResource>(
        stats_mr, std::move(memory_available)
    );
    auto envvars = rapidsmpf::config::get_environment_variables();
    envvars["num_streaming_threads"] = std::to_string(cmd_options.num_streaming_threads);
    auto options = rapidsmpf::config::Options(envvars);
    auto stats = std::make_shared<rapidsmpf::Statistics>(&stats_mr);
    {
        auto comm = rapidsmpf::ucxx::init_using_mpi(mpi_comm, options);
        auto progress =
            std::make_shared<rapidsmpf::ProgressThread>(comm->logger(), stats);
        auto ctx =
            std::make_shared<rapidsmpf::streaming::Context>(options, comm, br, stats);
        comm->logger().print(
            "Executor has ", ctx->executor()->thread_count(), " threads"
        );
        comm->logger().print("Executor has ", ctx->comm()->nranks(), " ranks");

        std::string output_path = cmd_options.output_file;
        std::vector<double> timings;
        for (int i = 0; i < 2; i++) {
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
                // [o_orderpriority, order_count]
                auto grouped_concatenated = ctx->create_channel();
                // [o_orderpriority, order_count]
                auto grouped_finalized = ctx->create_channel();
                // [o_orderpriority, order_count]
                auto sorted = ctx->create_channel();

                nodes.push_back(read_lineitem(
                    ctx,
                    lineitem,
                    4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));
                nodes.push_back(
                    filter_lineitem(ctx, lineitem, filtered_lineitem)
                );  // l_orderkey
                nodes.push_back(read_orders(
                    ctx,
                    order,
                    4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));
                // nodes.push_back(select_columns(ctx, order, projected_order, {1, 2}));

                nodes.push_back(
                    rapidsmpf::ndsh::shuffle(
                        ctx,
                        filtered_lineitem,
                        filtered_lineitem_shuffled,
                        {0},
                        cmd_options.num_partitions,
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                    )
                );

                if (cmd_options.use_shuffle_join) {
                    auto filtered_order_shuffled = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            order,
                            filtered_order_shuffled,
                            {0},
                            cmd_options.num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
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
                            order,
                            filtered_lineitem_shuffled,
                            orders_x_lineitem,
                            {0},
                            {0},
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::YES
                        )
                    );
                }

                nodes.push_back(
                    select_columns(ctx, orders_x_lineitem, projected_columns, {1})
                );

                nodes.push_back(
                    chunkwise_groupby_agg(ctx, projected_columns, grouped_chunkwise)
                );
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx,
                        grouped_chunkwise,
                        grouped_concatenated,
                        rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                    )
                );
                nodes.push_back(final_groupby_agg(
                    ctx,
                    grouped_concatenated,
                    grouped_finalized,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                ));
                nodes.push_back(sort_by(ctx, grouped_finalized, sorted));
                nodes.push_back(write_parquet(ctx, sorted, output_path));
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
            comm->logger().print(
                "Iteration ", i, " pipeline construction time [s]: ", pipeline.count()
            );
            comm->logger().print("Iteration ", i, " compute time [s]: ", compute.count());
            timings.push_back(pipeline.count());
            timings.push_back(compute.count());
            ctx->comm()->logger().print(stats->report());
            RAPIDSMPF_MPI(MPI_Barrier(mpi_comm));
        }
        if (comm->rank() == 0) {
            for (int i = 0; i < 2; i++) {
                comm->logger().print(
                    "Iteration ",
                    i,
                    " pipeline construction time [s]: ",
                    timings[size_t(2 * i)]
                );
                comm->logger().print(
                    "Iteration ", i, " compute time [s]: ", timings[size_t(2 * i + 1)]
                );
            }
        }
    }

    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
