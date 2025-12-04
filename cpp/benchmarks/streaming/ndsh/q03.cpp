/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <any>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/merge.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "concatenate.hpp"
#include "join.hpp"
#include "utils.hpp"

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

rapidsmpf::streaming::Node read_customer(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "customer")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"c_custkey"})  // 0
                       .build();
    auto filter_expr = [&]() -> std::unique_ptr<rapidsmpf::streaming::Filter> {
        auto stream = ctx->br()->stream_pool().get_stream();
        auto owner = new std::vector<std::any>;
        owner->push_back(std::make_shared<cudf::string_scalar>("BUILDING", true, stream));
        owner->push_back(
            std::make_shared<cudf::ast::literal>(
                *std::any_cast<std::shared_ptr<cudf::string_scalar>>(owner->at(0))
            )
        );
        owner->push_back(
            std::make_shared<cudf::ast::column_name_reference>("c_mktsegment")
        );
        owner->push_back(
            std::make_shared<cudf::ast::operation>(
                cudf::ast::ast_operator::EQUAL,
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
                           "l_orderkey",  // 0
                           "l_extendedprice",  // 1
                           "l_discount",  // 2
                       })
                       .build();
    auto filter_expr = [&]() -> std::unique_ptr<rapidsmpf::streaming::Filter> {
        auto stream = ctx->br()->stream_pool().get_stream();
        auto owner = new std::vector<std::any>;
        constexpr auto date = cuda::std::chrono::year_month_day(
            cuda::std::chrono::year(1995),
            cuda::std::chrono::month(3),
            cuda::std::chrono::day(15)
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
                cudf::ast::ast_operator::GREATER,
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

rapidsmpf::streaming::Node read_orders(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "orders")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({
                           "o_orderkey",  // 0
                           "o_orderdate",  // 1
                           "o_shippriority",  // 2
                           "o_custkey"  // 3
                       })
                       .build();
    auto filter_expr = [&]() -> std::unique_ptr<rapidsmpf::streaming::Filter> {
        auto stream = ctx->br()->stream_pool().get_stream();
        auto owner = new std::vector<std::any>;
        constexpr auto date = cuda::std::chrono::year_month_day(
            cuda::std::chrono::year(1995),
            cuda::std::chrono::month(3),
            cuda::std::chrono::day(15)
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
            std::make_shared<cudf::ast::column_name_reference>("o_orderdate")
        );
        owner->push_back(
            std::make_shared<cudf::ast::operation>(
                cudf::ast::ast_operator::LESS,
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

// In: [o_orderkey, o_orderdate, o_shippriority, revenue]
[[maybe_unused]] rapidsmpf::streaming::Node chunkwise_groupby_agg(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::vector<cudf::table> partial_results;
    std::uint64_t sequence = 0;
    co_await ctx->executor()->schedule();
    ctx->comm()->logger().print("Chunkwise groupby");
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();

        auto grouper = cudf::groupby::groupby(
            // group by [o_orderkey, o_orderdate, o_shippriority]
            table.select({0, 1, 2}),
            cudf::null_policy::EXCLUDE,
            cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(3), std::move(aggs))
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
            table.select({0, 1, 2}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(3), std::move(aggs))
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
            // We will only actually bother to do this on rank zero.
            auto result_view = global_result->view();
            auto grouper = cudf::groupby::groupby(
                result_view.select({0, 1, 2}),
                cudf::null_policy::EXCLUDE,
                cudf::sorted::NO
            );
            auto requests = std::vector<cudf::groupby::aggregation_request>();
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                cudf::groupby::aggregation_request(result_view.column(3), std::move(aggs))
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

// In: o_orderkey, o_orderdate, o_shippriority, l_extendedprice, l_discount
// Out: o_orderkey, o_orderdate, o_shippriority, revenue = (l_extendedprice - (1 -
// l_discount))
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
        std::vector<std::unique_ptr<cudf::column>> result;
        result.reserve(4);

        // o_orderkey
        result.push_back(
            std::make_unique<cudf::column>(
                table.column(0), chunk_stream, ctx->br()->device_mr()
            )
        );
        // o_orderdate
        result.push_back(
            std::make_unique<cudf::column>(
                table.column(1), chunk_stream, ctx->br()->device_mr()
            )
        );
        // o_shippriority
        result.push_back(
            std::make_unique<cudf::column>(
                table.column(2), chunk_stream, ctx->br()->device_mr()
            )
        );
        auto extendedprice = table.column(3);
        auto discount = table.column(4);
        std::string udf =
            R"***(
static __device__ void calculate_revenue(double *revenue, double extprice, double discount) {
    *revenue = extprice * (1 - discount);
}
           )***";

        // revenue
        result.push_back(
            cudf::transform(
                {extendedprice, discount},
                udf,
                cudf::data_type(cudf::type_id::FLOAT64),
                false,
                std::nullopt,
                cudf::null_aware::NO,
                chunk_stream,
                ctx->br()->device_mr()
            )
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

// take first 10 rows
[[maybe_unused]] rapidsmpf::streaming::Node top_k(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys,
    std::vector<cudf::order> order,
    cudf::size_type k
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    co_await ctx->executor()->schedule();
    std::vector<std::unique_ptr<cudf::table>> partials;
    std::vector<rmm::cuda_stream_view> chunk_streams;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto const indices = cudf::sorted_order(
            chunk.table_view().select(keys),
            order,
            {},
            chunk.stream(),
            ctx->br()->device_mr()
        );
        partials.push_back(
            cudf::gather(
                chunk.table_view(),
                cudf::split(indices->view(), {k}, chunk.stream()).front(),
                cudf::out_of_bounds_policy::DONT_CHECK,
                chunk.stream(),
                ctx->br()->device_mr()
            )
        );
        chunk_streams.push_back(chunk.stream());
    }

    // TODO:
    auto out_stream = chunk_streams.front();
    rapidsmpf::CudaEvent event;
    rapidsmpf::cuda_stream_join(
        std::ranges::single_view{out_stream}, chunk_streams, &event
    );
    std::vector<cudf::table_view> views;
    std::ranges::transform(partials, std::back_inserter(views), [](auto& t) {
        return t->view();
    });
    auto merged = cudf::merge(views, keys, order, {}, out_stream, ctx->br()->device_mr());
    auto result =
        std::make_unique<cudf::table>(cudf::slice(merged->view(), {0, 10}, out_stream));
    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(result), out_stream
            )
        )
    );
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
    // orderkey, revenue, orderdate, shippriority
    auto table = chunk.table_view().select({0, 3, 1, 2});
    auto builder = cudf::io::parquet_writer_options::builder(sink, table);
    auto metadata = cudf::io::table_input_metadata(table);
    metadata.column_metadata[0].set_name("l_orderkey");
    metadata.column_metadata[1].set_name("revenue");
    metadata.column_metadata[2].set_name("o_orderdate");
    metadata.column_metadata[3].set_name("o_shippriority");
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

int main(int argc, char** argv) {
    cudaFree(nullptr);
    auto mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_wrapper = rapidsmpf::RmmResourceAdaptor(&mr);
    auto arguments = rapidsmpf::ndsh::parse_arguments(argc, argv);
    auto ctx = rapidsmpf::ndsh::create_context(arguments, &stats_wrapper);
    std::string output_path = arguments.output_file;
    std::vector<double> timings;
    for (int i = 0; i < arguments.num_iterations; i++) {
        int op_id{0};
        std::vector<rapidsmpf::streaming::Node> nodes;
        auto start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q3 pipeline");
            auto customer = ctx->create_channel();
            auto lineitem = ctx->create_channel();
            auto orders = ctx->create_channel();

            auto customer_x_orders = ctx->create_channel();
            auto customer_x_orders_x_lineitem = ctx->create_channel();

            // Out: "c_custkey"
            nodes.push_back(read_customer(
                ctx,
                customer,
                /* num_tickets */ 2,
                arguments.num_rows_per_chunk,
                arguments.input_directory
            ));
            // Out: o_orderkey, o_orderdate, o_shippriority, o_custkey
            nodes.push_back(read_orders(
                ctx, orders, 6, arguments.num_rows_per_chunk, arguments.input_directory
            ));
            // join c_custkey = o_custkey
            // Out: o_orderkey, o_orderdate, o_shippriority
            nodes.push_back(
                rapidsmpf::ndsh::inner_join_broadcast(
                    ctx,
                    customer,
                    orders,
                    customer_x_orders,
                    {0},
                    {3},
                    static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                    rapidsmpf::ndsh::KeepKeys::NO
                )
            );
            // Out: l_orderkey, l_extendedprice, l_discount
            nodes.push_back(read_lineitem(
                ctx,
                lineitem,
                /* num_tickets */ 6,
                arguments.num_rows_per_chunk,
                arguments.input_directory
            ));

            // join o_orderkey = l_orderkey
            // Out: o_orderkey, o_orderdate, o_shippriority, l_extendedprice,
            // l_discount
            nodes.push_back(
                rapidsmpf::ndsh::inner_join_broadcast(
                    ctx,
                    customer_x_orders,
                    lineitem,
                    customer_x_orders_x_lineitem,
                    {0},
                    {0},
                    static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                    rapidsmpf::ndsh::KeepKeys::YES
                )
            );

            auto groupby_input = ctx->create_channel();
            // Out: o_orderkey, o_orderdate, o_shippriority, revenue
            nodes.push_back(select_columns_for_groupby(
                ctx, customer_x_orders_x_lineitem, groupby_input
            ));
            auto chunkwise_groupby_output = ctx->create_channel();
            // Out: o_orderkey, o_orderdate, o_shippriority, revenue
            nodes.push_back(
                chunkwise_groupby_agg(ctx, groupby_input, chunkwise_groupby_output)
            );
            auto concatenated_groupby_output = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    chunkwise_groupby_output,
                    concatenated_groupby_output,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );
            auto groupby_output = ctx->create_channel();
            // Out: o_orderkey, o_orderdate, o_shippriority, revenue
            nodes.push_back(final_groupby_agg(
                ctx,
                concatenated_groupby_output,
                groupby_output,
                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
            ));
            auto topk = ctx->create_channel();
            // Out: o_orderkey, o_orderdate, o_shippriority, revenue
            nodes.push_back(top_k(
                ctx,
                groupby_output,
                topk,
                {3, 1},
                {cudf::order::DESCENDING, cudf::order::ASCENDING},
                10
            ));

            nodes.push_back(write_parquet(ctx, topk, output_path));
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
        timings.push_back(pipeline.count());
        timings.push_back(compute.count());
        ctx->comm()->logger().print(ctx->statistics()->report());
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

    if (rapidsmpf::mpi::is_initialized()) {
        RAPIDSMPF_MPI(MPI_Finalize());
    }
    return 0;
}
