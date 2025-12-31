/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <any>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/context.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
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

#include "concatenate.hpp"
#include "groupby.hpp"
#include "join.hpp"
#include "parquet_writer.hpp"
#include "sort.hpp"
#include "utils.hpp"

namespace {

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
                           "l_returnflag",  // 0
                           "l_linestatus",  // 1
                           "l_quantity",  // 2
                           "l_extendedprice",  // 3
                           "l_discount",  // 4
                           "l_tax"  // 5
                       })
                       .build();
    // TODO: utility to get logical types from parquet.
    using timestamp_type = cudf::timestamp_ms;
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
            std::make_shared<cudf::timestamp_scalar<timestamp_type>>(
                sys_days.time_since_epoch(), true, stream
            )
        );
        owner->push_back(
            std::make_shared<cudf::ast::literal>(
                *std::any_cast<std::shared_ptr<cudf::timestamp_scalar<timestamp_type>>>(
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

std::vector<rapidsmpf::ndsh::groupby_request> chunkwise_groupby_requests() {
    auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
    std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
    // sum(l_quantity), sum(l_extendedprice), sum(disc_price), sum(charge),
    // sum(l_discount)
    for (cudf::size_type idx = 2; idx < 7; idx++) {
        aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
        requests.emplace_back(idx, std::move(aggs));
    }
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
    // sum(l_quantity), sum(l_extendedprice), sum(disc_price), sum(charge),
    // sum(l_discount), sum(count(*))
    for (cudf::size_type idx = 2; idx < 8; idx++) {
        aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
        requests.emplace_back(idx, std::move(aggs));
    }
    return requests;
}

rapidsmpf::streaming::Node postprocess_group_by(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    RAPIDSMPF_EXPECTS(
        (co_await ch_in->receive()).empty(), "Expecting concatenated input at this point"
    );
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto stream = chunk.stream();
    auto columns =
        cudf::table{chunk.table_view(), stream, ctx->br()->device_mr()}.release();
    std::ignore = std::move(chunk);
    auto count = std::move(columns.back());
    columns.pop_back();
    auto discount = std::move(columns.back());
    columns.pop_back();
    for (std::size_t i = 2; i < 4; i++) {
        columns.push_back(
            cudf::binary_operation(
                columns[i]->view(),
                count->view(),
                cudf::binary_operator::TRUE_DIV,
                cudf::data_type(cudf::type_id::FLOAT64),
                stream,
                ctx->br()->device_mr()
            )
        );
    }
    columns.push_back(
        cudf::binary_operation(
            discount->view(),
            count->view(),
            cudf::binary_operator::TRUE_DIV,
            cudf::data_type(cudf::type_id::FLOAT64),
            stream,
            ctx->br()->device_mr()
        )
    );
    columns.push_back(std::move(count));
    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            msg.sequence_number(),
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::make_unique<cudf::table>(std::move(columns)), stream
            )
        )
    );
    co_await ch_out->drain(ctx->executor());
}

// In: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// l_discount, l_tax
// Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// disc_price = (l_extendedprice * (1 - l_discount)),
// charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax)),
// l_discount
rapidsmpf::streaming::Node select_columns_for_groupby(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    co_await ctx->executor()->schedule();
    while (!ch_out->is_shutdown()) {
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
                cudf::output_nullability::PRESERVE,
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
                cudf::output_nullability::PRESERVE,
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
}  // namespace

/**
 * @brief Run a derived version of TPC-H query 1.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     l_returnflag,
 *     l_linestatus,
 *     sum(l_quantity) as sum_qty,
 *     sum(l_extendedprice) as sum_base_price,
 *     sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
 *     sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
 *     avg(l_quantity) as avg_qty,
 *     avg(l_extendedprice) as avg_price,
 *     avg(l_discount) as avg_disc,
 *     count(*) as count_order
 * from
 *     lineitem
 * where
 *     l_shipdate <= DATE '1998-09-02'
 * group by
 *     l_returnflag,
 *     l_linestatus
 * order by
 *     l_returnflag,
 *     l_linestatus
 * @endcode{}
 */
int main(int argc, char** argv) {
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
    for (int i = 0; i < arguments.num_iterations; i++) {
        int op_id = 0;
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
                arguments.num_rows_per_chunk,
                arguments.input_directory
            ));

            auto chunkwise_groupby_input = ctx->create_channel();
            // Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
            // disc_price = (l_extendedprice * (1 - l_discount)),
            // charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax))
            // l_discount
            nodes.push_back(
                select_columns_for_groupby(ctx, lineitem, chunkwise_groupby_input)
            );
            auto chunkwise_groupby_output = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::chunkwise_group_by(
                    ctx,
                    chunkwise_groupby_input,
                    chunkwise_groupby_output,
                    {0, 1},
                    chunkwise_groupby_requests(),
                    cudf::null_policy::INCLUDE
                )
            );
            auto final_groupby_input = ctx->create_channel();
            if (ctx->comm()->nranks() > 1) {
                nodes.push_back(
                    rapidsmpf::ndsh::broadcast(
                        ctx,
                        chunkwise_groupby_output,
                        final_groupby_input,
                        static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                        rapidsmpf::streaming::AllGather::Ordered::NO
                    )
                );
            } else {
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx, chunkwise_groupby_output, final_groupby_input
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
                        {0, 1},
                        final_groupby_requests(),
                        cudf::null_policy::INCLUDE
                    )
                );
                auto sorted_input = ctx->create_channel();
                nodes.push_back(
                    postprocess_group_by(ctx, final_groupby_output, sorted_input)
                );
                auto sorted_output = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::chunkwise_sort_by(
                        ctx,
                        sorted_input,
                        sorted_output,
                        {0, 1},
                        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                        {cudf::order::ASCENDING, cudf::order::ASCENDING},
                        {cudf::null_order::BEFORE, cudf::null_order::BEFORE}
                    )
                );
                nodes.push_back(
                    rapidsmpf::ndsh::write_parquet(
                        ctx,
                        sorted_output,
                        cudf::io::sink_info(output_path),
                        {"l_returnflag",
                         "l_linestatus",
                         "sum_qty",
                         "sum_base_price",
                         "sum_disc_price",
                         "sum_charge",
                         "avg_qty",
                         "avg_price",
                         "avg_disc",
                         "count_order"}

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
            RAPIDSMPF_NVTX_SCOPED_RANGE("Q1 Iteration");
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
