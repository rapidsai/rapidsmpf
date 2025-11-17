/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <any>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/merge.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/single.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
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
#include "rapidsmpf/cuda_stream.hpp"
#include "rapidsmpf/owning_wrapper.hpp"
#include "rapidsmpf/streaming/core/coro_utils.hpp"
#include "utilities.hpp"

// select
//     l_orderkey,
//     sum(l_extendedprice * (1 - l_discount)) as revenue,
//     o_orderdate,
//     o_shippriority
// from
//     customer,
//     orders,
//     lineitem
// where
//     c_mktsegment = 'BUILDING'
//     and c_custkey = o_custkey
//     and l_orderkey = o_orderkey
//     and o_orderdate < '1995-03-15'
//     and l_shipdate > '1995-03-15'
// group by
//     l_orderkey,
//     o_orderdate,
//     o_shippriority
// order by
//     revenue desc,
//     o_orderdate
// limit 10

namespace {

std::string get_table_path(
    std::string const& input_directory, std::string const& table_name
) {
    auto dir = input_directory.empty() ? "." : input_directory;
    auto file_path = dir + "/" + table_name + ".parquet";

    if (std::filesystem::exists(file_path)) {
        return file_path;
    }

    return dir + "/" + table_name + "/";
}

[[maybe_unused]] rapidsmpf::streaming::Node read_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "lineitem")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({
                           "l_returnflag",  // 0
                           "l_linestatus",  // 1
                           "l_quantity",  // 2
                           "l_extendedprice",  // 3
                           "l_discount",  // 4
                           "l_tax"  // 5
                       })
                       .build();
    auto filter_expr = [&]() -> std::unique_ptr<rapidsmpf::streaming::Filter> {
        auto stream = ctx->br()->stream_pool().get_stream();
        auto owner = new std::vector<std::any>;
        constexpr auto date = cuda::std::chrono::year_month_day(
            cuda::std::chrono::year(1998),
            cuda::std::chrono::month(9),
            cuda::std::chrono::day(2)
        );
        auto sys_days = cuda::std::chrono::sys_days(date);
        owner->push_back(
            std::make_shared<cudf::timestamp_scalar<cudf::timestamp_D>>(
                sys_days, true, stream
            )
        );
        owner->push_back(
            std::make_shared<cudf::ast::literal>(
                *std::any_cast<
                    std::shared_ptr<cudf::timestamp_scalar<cudf::timestamp_D>>>(
                    owner->at(0)
                )
            )
        );
        owner->push_back(
            std::make_shared<cudf::ast::column_name_reference>("l_shipdate")
        );
        owner->push_back(
            std::make_shared<cudf::ast::operation>(
                cudf::ast::ast_operator::LESS_EQUAL,
                *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(
                    owner->at(2)
                ),
                *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(1))
            )
        );
        return std::make_unique<rapidsmpf::streaming::Filter>(
            stream,
            *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
            rapidsmpf::OwningWrapper(static_cast<void*>(owner), [](void* p) {
                delete static_cast<std::vector<std::any>*>(p);
            })
        );
    }();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter_expr)
    );
}

// l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// disc_price = (l_extendedprice * (1 - l_discount)),
// charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax))
// l_discount
[[maybe_unused]] rapidsmpf::streaming::Node chunkwise_groupby_agg(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::vector<cudf::table> partial_results;
    std::uint64_t sequence = 0;
    ctx->comm()->logger().print("Chunkwise groupby");
    auto grouper = [&]() -> coro::task<void> {
        while (true) {
            auto msg = co_await ch_in->receive();
            co_await ctx->executor()->schedule();
            if (msg.empty()) {
                break;
            }
            auto chunk = rapidsmpf::ndsh::to_device(
                ctx, msg.release<rapidsmpf::streaming::TableChunk>()
            );
            auto chunk_stream = chunk.stream();
            auto table = chunk.table_view();

            auto grouper = cudf::groupby::groupby(
                // group by [l_returnflag, l_linestatus]
                table.select({0, 1}),
                cudf::null_policy::EXCLUDE,
                cudf::sorted::NO
            );
            auto requests = std::vector<cudf::groupby::aggregation_request>();
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(l_quantity)
                cudf::groupby::aggregation_request(table.column(2), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(l_extendedprice)
                cudf::groupby::aggregation_request(table.column(3), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(disc_price)
                cudf::groupby::aggregation_request(table.column(4), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(charge)
                cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(l_discount)
                cudf::groupby::aggregation_request(table.column(6), std::move(aggs))
            );
            aggs.push_back(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(
                    cudf::null_policy::INCLUDE
                )
            );
            requests.push_back(
                // count(*)
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
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    sequence++,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::make_unique<cudf::table>(std::move(result)), chunk_stream
                    )
                )
            );
        }
    };
    rapidsmpf::streaming::coro_results(
        co_await coro::when_all(grouper(), grouper(), grouper(), grouper())
    );
    co_await ch_out->drain(ctx->executor());
}

[[maybe_unused]] rapidsmpf::streaming::Node final_groupby_agg(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    // TODO: requires concatenated input stream.
    auto msg = co_await ch_in->receive();
    auto next = co_await ch_in->receive();
    ctx->comm()->logger().print("Final groupby");
    RAPIDSMPF_EXPECTS(next.empty(), "Expecting concatenated input at this point");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();
    std::unique_ptr<cudf::table> local_result{nullptr};
    if (!table.is_empty()) {
        auto grouper = cudf::groupby::groupby(
            table.select({0, 1}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            // sum(l_quantity)
            cudf::groupby::aggregation_request(table.column(2), std::move(aggs))
        );
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            // sum(l_extendedprice)
            cudf::groupby::aggregation_request(table.column(3), std::move(aggs))
        );
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            // sum(disc_price)
            cudf::groupby::aggregation_request(table.column(4), std::move(aggs))
        );
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            // sum(charge)
            cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
        );
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            // sum(l_discount)
            cudf::groupby::aggregation_request(table.column(6), std::move(aggs))
        );
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            // sum(count(*))
            cudf::groupby::aggregation_request(table.column(7), std::move(aggs))
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
            std::vector<rapidsmpf::PackedData> chunks;
            chunks.reserve(packed_data.size());
            std::ranges::transform(
                packed_data, std::back_inserter(chunks), [](auto& chunk) {
                    return std::move(chunk.data);
                }
            );
            auto global_result = rapidsmpf::unpack_and_concat(
                rapidsmpf::unspill_partitions(
                    std::move(chunks), ctx->br(), true, ctx->statistics()
                ),
                chunk_stream,
                ctx->br(),
                ctx->statistics()
            );
            auto table = global_result->view();
            auto grouper = cudf::groupby::groupby(
                table.select({0, 1}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
            );
            auto requests = std::vector<cudf::groupby::aggregation_request>();
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(l_quantity)
                cudf::groupby::aggregation_request(table.column(2), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(l_extendedprice)
                cudf::groupby::aggregation_request(table.column(3), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(disc_price)
                cudf::groupby::aggregation_request(table.column(4), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(charge)
                cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(l_discount)
                cudf::groupby::aggregation_request(table.column(6), std::move(aggs))
            );
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                // sum(count(*))
                cudf::groupby::aggregation_request(table.column(7), std::move(aggs))
            );
            auto [keys, results] =
                grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
            // Drop chunk, we don't need it.
            std::ignore = std::move(chunk);
            auto result = keys->release();
            for (auto&& r : results) {
                std::ranges::move(r.results, std::back_inserter(result));
            }
            auto count = std::move(result.back());
            result.pop_back();
            auto discount = std::move(result.back());
            result.pop_back();
            for (std::size_t i = 2; i < 4; i++) {
                result.push_back(
                    cudf::binary_operation(
                        result[i]->view(),
                        count->view(),
                        cudf::binary_operator::TRUE_DIV,
                        cudf::data_type(cudf::type_id::FLOAT64),
                        chunk_stream,
                        ctx->br()->device_mr()
                    )
                );
            }
            result.push_back(
                cudf::binary_operation(
                    discount->view(),
                    count->view(),
                    cudf::binary_operator::TRUE_DIV,
                    cudf::data_type(cudf::type_id::FLOAT64),
                    chunk_stream,
                    ctx->br()->device_mr()
                )
            );

            result.push_back(std::move(count));
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::make_unique<cudf::table>(std::move(result)), chunk_stream
                    )
                )
            );
        } else {
            std::ignore = std::move(packed_data);
        }
    } else {
        auto result = local_result->release();
        auto count = std::move(result.back());
        result.pop_back();
        auto discount = std::move(result.back());
        result.pop_back();
        for (std::size_t i = 2; i < 4; i++) {
            result.push_back(
                cudf::binary_operation(
                    result[i]->view(),
                    count->view(),
                    cudf::binary_operator::TRUE_DIV,
                    cudf::data_type(cudf::type_id::FLOAT64),
                    chunk_stream,
                    ctx->br()->device_mr()
                )
            );
        }
        result.push_back(
            cudf::binary_operation(
                discount->view(),
                count->view(),
                cudf::binary_operator::TRUE_DIV,
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            )
        );
        result.push_back(std::move(count));
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(std::move(result)), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

// In: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// l_discount, l_tax
// Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// disc_price = (l_extendedprice * (1 - l_discount)),
// charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax)),
// l_discount
[[maybe_unused]] rapidsmpf::streaming::Node select_columns_for_groupby(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto sequence_number = msg.sequence_number();
        auto table = chunk.table_view();
        // l_returnflag, l_linestatus, l_quantity, l_extendedprice
        auto result =
            cudf::table(table.select({0, 1, 2, 3}), chunk_stream, ctx->br()->device_mr())
                .release();
        result.reserve(7);
        auto extendedprice = table.column(3);
        auto discount = table.column(4);
        auto tax = table.column(5);
        std::string udf_disc_price =
            R"***(
static __device__ void calculate_disc_price(double *disc_price, double extprice, double discount) {
    *disc_price = extprice * (1 - discount);
}
           )***";
        std::string udf_charge =
            R"***(
static __device__ void calculate_charge(double *charge, double discprice, double tax) {
    *charge = discprice * (1 + tax);
}
           )***";

        // disc_price
        result.push_back(
            cudf::transform(
                {extendedprice, discount},
                udf_disc_price,
                cudf::data_type(cudf::type_id::FLOAT64),
                false,
                std::nullopt,
                cudf::null_aware::NO,
                chunk_stream,
                ctx->br()->device_mr()
            )
        );
        // charge
        result.push_back(
            cudf::transform(
                {result.back()->view(), tax},
                udf_charge,
                cudf::data_type(cudf::type_id::FLOAT64),
                false,
                std::nullopt,
                cudf::null_aware::NO,
                chunk_stream,
                ctx->br()->device_mr()
            )
        );
        // l_discount
        result.push_back(
            std::make_unique<cudf::column>(discount, chunk_stream, ctx->br()->device_mr())
        );
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence_number,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(std::move(result)), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

[[maybe_unused]] rapidsmpf::streaming::Node sort_by(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    // We know we only have a single chunk from the groupby
    if (msg.empty()) {
        co_return;
    }
    ctx->comm()->logger().print("Sortby");
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto table = chunk.table_view();
    auto result = rapidsmpf::streaming::to_message(
        0,
        std::make_unique<rapidsmpf::streaming::TableChunk>(
            cudf::sort_by_key(
                table,
                table.select({0, 1}),
                {cudf::order::ASCENDING, cudf::order::ASCENDING},
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

// In: o_orderkey, o_orderdate, o_shippriority, revenue
[[maybe_unused]] rapidsmpf::streaming::Node write_parquet(
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
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto sink = cudf::io::sink_info(output_path);
    auto table = chunk.table_view();
    auto builder = cudf::io::parquet_writer_options::builder(sink, table);
    auto metadata = cudf::io::table_input_metadata(table);
    metadata.column_metadata[0].set_name("l_returnflag");
    metadata.column_metadata[1].set_name("l_linestatus");
    metadata.column_metadata[2].set_name("sum_qty");
    metadata.column_metadata[3].set_name("sum_base_price");
    metadata.column_metadata[4].set_name("sum_disc_price");
    metadata.column_metadata[5].set_name("sum_charge");
    metadata.column_metadata[6].set_name("avg_qty");
    metadata.column_metadata[7].set_name("avg_price");
    metadata.column_metadata[8].set_name("avg_disc");
    metadata.column_metadata[9].set_name("count_order");
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

[[maybe_unused]] rapidsmpf::streaming::Node consume(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        ctx->comm()->logger().print(
            "Consumed chunk with ",
            chunk.table_view().num_rows(),
            " rows and ",
            chunk.table_view().num_columns(),
            " columns"
        );
    }
}
}  // namespace

struct ProgramOptions {
    int num_streaming_threads{1};
    int num_iterations{2};
    cudf::size_type num_rows_per_chunk{100'000'000};
    std::optional<double> spill_device_limit{std::nullopt};
    bool use_shuffle_join = false;
    std::string output_file;
    std::string input_directory;
};

ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;

    auto print_usage = [&argv]() {
        std::cerr
            << "Usage: " << argv[0] << " [options]\n"
            << "Options:\n"
            << "  --num-streaming-threads <n>  Number of streaming threads (default: 1)\n"
            << "  --num-iterations <n>         Number of iterations (default: 2)\n"
            << "  --num-rows-per-chunk <n>     Number of rows per chunk (default: "
               "100000000)\n"
            << "  --spill-device-limit <n>     Fractional spill device limit (default: "
               "None)\n"
            << "  --use-shuffle-join           Use shuffle join (default: false)\n"
            << "  --output-file <path>         Output file path (required)\n"
            << "  --input-directory <path>     Input directory path (required)\n"
            << "  --help                       Show this help message\n";
    };

    static struct option long_options[] = {
        {"num-streaming-threads", required_argument, nullptr, 1},
        {"num-rows-per-chunk", required_argument, nullptr, 2},
        {"use-shuffle-join", no_argument, nullptr, 3},
        {"output-file", required_argument, nullptr, 4},
        {"input-directory", required_argument, nullptr, 5},
        {"help", no_argument, nullptr, 6},
        {"spill-device-limit", required_argument, nullptr, 7},
        {"num-iterations", required_argument, nullptr, 8},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    int option_index = 0;

    bool saw_output_file = false;
    bool saw_input_directory = false;

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
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
            print_usage();
            std::exit(0);
        case 7:
            options.spill_device_limit = std::stod(optarg);
            break;
        case 8:
            options.num_iterations = std::atoi(optarg);
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
    // rmm::mr::cuda_memory_resource base{};
    // rmm::mr::pool_memory_resource mr{&base, pool_size};
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
        [[maybe_unused]] int op_id = 0;
        for (int i = 0; i < cmd_options.num_iterations; i++) {
            std::vector<rapidsmpf::streaming::Node> nodes;
            auto start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q1 pipeline");

                // Input data channels
                auto lineitem = ctx->create_channel();
                // Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
                // l_discount, l_tax
                nodes.push_back(read_lineitem(
                    ctx,
                    lineitem,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));

                auto groupby_input = ctx->create_channel();
                // Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
                // disc_price = (l_extendedprice * (1 - l_discount)),
                // charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax))
                // l_discount
                nodes.push_back(select_columns_for_groupby(ctx, lineitem, groupby_input));
                auto chunkwise_groupby = ctx->create_channel();
                nodes.push_back(
                    chunkwise_groupby_agg(ctx, groupby_input, chunkwise_groupby)
                );
                auto final_groupby_input = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx, chunkwise_groupby, final_groupby_input
                    )
                );
                auto groupby_output = ctx->create_channel();
                nodes.push_back(final_groupby_agg(
                    ctx,
                    final_groupby_input,
                    groupby_output,
                    static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                ));
                auto sorted = ctx->create_channel();
                nodes.push_back(sort_by(ctx, groupby_output, sorted));
                nodes.push_back(write_parquet(ctx, sorted, output_path));
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> pipeline = end - start;
            start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Q3 Iteration");
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
            for (int i = 0; i < cmd_options.num_iterations; i++) {
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
