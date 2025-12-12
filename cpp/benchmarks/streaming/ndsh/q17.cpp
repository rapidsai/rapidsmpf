/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/reduction.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
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
#include <rapidsmpf/streaming/core/fanout.hpp>
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

rapidsmpf::streaming::Node read_lineitem(
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
                       .columns({"l_partkey", "l_quantity", "l_extendedprice"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_part(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "part")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"p_partkey"})
                       .build();

    // Build the filter expression: p_brand = 'Brand#23' AND p_container = 'MED BOX'
    auto owner = new std::vector<std::any>;
    auto filter_stream = ctx->br()->stream_pool().get_stream();

    // 0, 1: column references
    owner->push_back(std::make_shared<cudf::ast::column_name_reference>("p_brand"));
    owner->push_back(std::make_shared<cudf::ast::column_name_reference>("p_container"));

    // 2, 3: string_scalars
    owner->push_back(
        std::make_shared<cudf::string_scalar>("Brand#23", true, filter_stream)
    );
    owner->push_back(
        std::make_shared<cudf::string_scalar>("MED BOX", true, filter_stream)
    );

    // 4, 5: literals
    owner->push_back(
        std::make_shared<cudf::ast::literal>(
            *std::any_cast<std::shared_ptr<cudf::string_scalar>>(owner->at(2))
        )
    );
    owner->push_back(
        std::make_shared<cudf::ast::literal>(
            *std::any_cast<std::shared_ptr<cudf::string_scalar>>(owner->at(3))
        )
    );

    // 6: operation (EQUAL, p_brand, "Brand#23")
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            cudf::ast::ast_operator::EQUAL,
            *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(
                owner->at(0)
            ),
            *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(4))
        )
    );

    // 7: operation (EQUAL, p_container, "MED BOX")
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            cudf::ast::ast_operator::EQUAL,
            *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(
                owner->at(1)
            ),
            *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(5))
        )
    );

    // 8: operation (LOGICAL_AND, brand_eq, container_eq)
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            cudf::ast::ast_operator::LOGICAL_AND,
            *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->at(6)),
            *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->at(7))
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

// Node to compute sum and count of quantity per partkey
rapidsmpf::streaming::Node compute_avg_quantity(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::uint64_t sequence = 0;
    while (true) {
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

        // table has: p_partkey, l_quantity, l_extendedprice
        // Group by p_partkey and compute sum and count of l_quantity
        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        aggs.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());

        // Output: p_partkey, sum(l_quantity), count(l_quantity)
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

template <typename T>
std::unique_ptr<cudf::column> column_from_value(
    const T& value, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr
) {
    rmm::device_uvector<T> vec(1, stream, mr);
    vec.set_element_async(0, value, stream);

    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_to_id<T>()}, 1, vec.release(), rmm::device_buffer{}, 0
    );
}

// Final aggregation after the second join and filter
rapidsmpf::streaming::Node final_aggregation(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();

    // Process chunks incrementally to avoid OOM
    double local_sum = 0.0;
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

        if (!table.is_empty() && table.num_columns() >= 4) {
            // table has: key, avg_quantity, l_quantity, l_extendedprice
            // Filter: l_quantity < avg_quantity
            auto l_quantity = table.column(2);
            auto avg_quantity = table.column(1);
            auto filter_mask = cudf::binary_operation(
                l_quantity,
                avg_quantity,
                cudf::binary_operator::LESS,
                cudf::data_type(cudf::type_id::BOOL8),
                chunk_stream,
                ctx->br()->device_mr()
            );
            auto filtered = cudf::apply_boolean_mask(
                table, filter_mask->view(), chunk_stream, ctx->br()->device_mr()
            );

            // Sum l_extendedprice for this chunk
            if (filtered->num_rows() > 0) {
                auto l_extendedprice = filtered->view().column(3);
                auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
                auto sum_result = cudf::reduce(
                    l_extendedprice,
                    *sum_agg,
                    cudf::data_type(cudf::type_id::FLOAT64),
                    chunk_stream,
                    ctx->br()->device_mr()
                );

                // Accumulate the sum
                auto chunk_sum = static_cast<cudf::numeric_scalar<double>&>(*sum_result)
                                     .value(chunk_stream);
                local_sum += chunk_sum;
            }
        }
    }

    // Create result table with local sum
    auto chunk_stream = rmm::cuda_stream_view{};

    if (ctx->comm()->nranks() > 1) {
        std::unique_ptr<cudf::table> local_result{nullptr};
        if (local_sum > 0.0) {
            std::vector<std::unique_ptr<cudf::column>> columns;
            columns.push_back(
                column_from_value(local_sum, chunk_stream, ctx->br()->device_mr())
            );
            local_result = std::make_unique<cudf::table>(std::move(columns));
        }

        // Gather results from all ranks
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

            // Sum all the partial sums, THEN divide by 7.0
            if (global_result && global_result->num_rows() > 0) {
                auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
                auto sum_result = cudf::reduce(
                    global_result->view().column(0),
                    *sum_agg,
                    cudf::data_type(cudf::type_id::FLOAT64),
                    chunk_stream,
                    ctx->br()->device_mr()
                );

                // Now divide by 7.0
                auto total_sum = static_cast<cudf::numeric_scalar<double>&>(*sum_result)
                                     .value(chunk_stream);
                auto avg_yearly_val = total_sum / 7.0;

                std::vector<std::unique_ptr<cudf::column>> cols1;
                cols1.push_back(column_from_value(
                    avg_yearly_val, chunk_stream, ctx->br()->device_mr()
                ));
                co_await ch_out->send(
                    rapidsmpf::streaming::to_message(
                        0,
                        std::make_unique<rapidsmpf::streaming::TableChunk>(
                            std::make_unique<cudf::table>(std::move(cols1)), chunk_stream
                        )
                    )
                );
            } else {
                // No data after filtering - send empty result with 0.0
                std::vector<std::unique_ptr<cudf::column>> cols2;
                cols2.push_back(
                    column_from_value(0.0, chunk_stream, ctx->br()->device_mr())
                );
                co_await ch_out->send(
                    rapidsmpf::streaming::to_message(
                        0,
                        std::make_unique<rapidsmpf::streaming::TableChunk>(
                            std::make_unique<cudf::table>(std::move(cols2)), chunk_stream
                        )
                    )
                );
            }
        }
        // Non-zero ranks don't send anything (following q09 pattern)
    } else {
        // Single rank: divide by 7.0 here

        auto avg_yearly_val = local_sum / 7.0;

        std::vector<std::unique_ptr<cudf::column>> cols3;
        cols3.push_back(
            column_from_value(avg_yearly_val, chunk_stream, ctx->br()->device_mr())
        );
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(std::move(cols3)), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node round_result(
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
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto table = chunk.table_view();
    auto rounded = cudf::round(
        table.column(0),
        2,
        cudf::rounding_method::HALF_EVEN,
        chunk.stream(),
        ctx->br()->device_mr()
    );

    std::vector<std::unique_ptr<cudf::column>> result_cols;
    result_cols.push_back(std::move(rounded));

    auto result = rapidsmpf::streaming::to_message(
        0,
        std::make_unique<rapidsmpf::streaming::TableChunk>(
            std::make_unique<cudf::table>(std::move(result_cols)), chunk.stream()
        )
    );
    co_await ch_out->send(std::move(result));
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node groupby_avg_quantity(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    auto next = co_await ch_in->receive();
    RAPIDSMPF_EXPECTS(next.empty(), "Expecting concatenated input");

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();

    std::unique_ptr<cudf::table> local_result{nullptr};
    if (!table.is_empty()) {
        // table has: p_partkey, sum(l_quantity), count(l_quantity)
        // Group by p_partkey and sum the sums and counts
        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> sum_aggs1;
        sum_aggs1.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> sum_aggs2;
        sum_aggs2.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(1), std::move(sum_aggs1))
        );
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(2), std::move(sum_aggs2))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());

        // Output: p_partkey (as key), sum(l_quantity),
        // count(l_quantity) Don't compute avg here - do it after
        // global aggregation
        auto result = keys->release();
        result.push_back(std::move(results[0].results[0]));  // sum
        result.push_back(std::move(results[1].results[0]));  // count
        local_result = std::make_unique<cudf::table>(std::move(result));
    }

    if (ctx->comm()->nranks() > 1) {
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

            auto result_view = global_result->view();
            // result_view has: p_partkey, sum, count
            // Group by p_partkey and sum both the sums and counts
            auto grouper = cudf::groupby::groupby(
                result_view.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
            );
            auto requests = std::vector<cudf::groupby::aggregation_request>();
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> sum_aggs1;
            sum_aggs1.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> sum_aggs2;
            sum_aggs2.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            requests.push_back(
                cudf::groupby::aggregation_request(
                    result_view.column(1), std::move(sum_aggs1)
                )
            );
            requests.push_back(
                cudf::groupby::aggregation_request(
                    result_view.column(2), std::move(sum_aggs2)
                )
            );
            auto [keys, results] =
                grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());
            global_result.reset();

            // Compute mean = sum / count, then multiply by 0.2
            auto sum_col = results[0].results[0]->view();
            auto count_col = results[1].results[0]->view();
            auto mean_col = cudf::binary_operation(
                sum_col,
                count_col,
                cudf::binary_operator::DIV,
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            );

            auto scalar_02 = cudf::make_numeric_scalar(
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            );
            static_cast<cudf::numeric_scalar<double>*>(scalar_02.get())
                ->set_value(0.2, chunk_stream);
            auto avg_quantity = cudf::binary_operation(
                mean_col->view(),
                *scalar_02,
                cudf::binary_operator::MUL,
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            );

            auto result = keys->release();
            result.push_back(std::move(avg_quantity));
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::make_unique<cudf::table>(std::move(result)), chunk_stream
                    )
                )
            );
        } else {
            // Non-zero ranks: send empty table with correct schema
            // Schema: key (INT64), avg_quantity (FLOAT64)
            std::vector<std::unique_ptr<cudf::column>> empty_cols;
            empty_cols.push_back(
                cudf::make_empty_column(cudf::data_type(cudf::type_id::INT64))
            );
            empty_cols.push_back(
                cudf::make_empty_column(cudf::data_type(cudf::type_id::FLOAT64))
            );
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::make_unique<cudf::table>(std::move(empty_cols)), chunk_stream
                    )
                )
            );
        }
    } else {
        if (local_result) {
            // Single-rank: need to compute avg = 0.2 * (sum / count) just like multi-rank
            // local_result has: p_partkey, sum(l_quantity), count(l_quantity)
            auto result_view = local_result->view();
            auto sum_col = result_view.column(1);
            auto count_col = result_view.column(2);

            // Compute mean = sum / count
            auto mean_col = cudf::binary_operation(
                sum_col,
                count_col,
                cudf::binary_operator::DIV,
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            );

            // Multiply by 0.2
            auto scalar_02 = cudf::make_numeric_scalar(
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            );
            static_cast<cudf::numeric_scalar<double>*>(scalar_02.get())
                ->set_value(0.2, chunk_stream);
            auto avg_quantity = cudf::binary_operation(
                mean_col->view(),
                *scalar_02,
                cudf::binary_operator::MUL,
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                ctx->br()->device_mr()
            );

            // Output: p_partkey (as key), avg_quantity
            std::vector<std::unique_ptr<cudf::column>> result;
            result.push_back(
                std::make_unique<cudf::column>(
                    result_view.column(0), chunk_stream, ctx->br()->device_mr()
                )
            );
            result.push_back(std::move(avg_quantity));

            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::make_unique<cudf::table>(std::move(result)), chunk_stream
                    )
                )
            );
        } else {
            // Single rank with no data: send empty table with correct
            // schema (key, avg_quantity)
            std::vector<std::unique_ptr<cudf::column>> empty_cols;
            empty_cols.push_back(
                cudf::make_empty_column(cudf::data_type(cudf::type_id::INT64))
            );
            empty_cols.push_back(
                cudf::make_empty_column(cudf::data_type(cudf::type_id::FLOAT64))
            );
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::make_unique<cudf::table>(std::move(empty_cols)), chunk_stream
                    )
                )
            );
        }
    }
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
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto sink = cudf::io::sink_info(output_path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("avg_yearly");
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
};

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

/**
 * @brief Run a derived version of TPC-H query 17.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     round(sum(l_extendedprice) / 7.0, 2) as avg_yearly
 * from
 *     lineitem,
 *     part
 * where
 *     p_partkey = l_partkey
 *     and p_brand = 'Brand#23'
 *     and p_container = 'MED BOX'
 *     and l_quantity < (
 *         select
 *             0.2 * avg(l_quantity)
 *         from
 *             lineitem
 *         where
 *             l_partkey = p_partkey
 *     )
 * @endcode{}
 */
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
                RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q17 pipeline");

                // Read part with filter pushed down, then project to just p_partkey
                auto part = ctx->create_channel();
                nodes.push_back(read_part(
                    ctx,
                    part,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // p_partkey (filtered)

                // Read lineitem
                auto lineitem = ctx->create_channel();
                nodes.push_back(read_lineitem(
                    ctx,
                    lineitem,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // l_partkey, l_quantity, l_extendedprice

                // Inner join: part x lineitem on p_partkey = l_partkey
                auto part_x_lineitem = ctx->create_channel();
                if (cmd_options.use_shuffle_join) {
                    auto projected_part_shuffled = ctx->create_channel();
                    auto lineitem_shuffled = ctx->create_channel();
                    std::uint32_t num_partitions = 16;
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            part,
                            projected_part_shuffled,
                            {0},
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            lineitem,
                            lineitem_shuffled,
                            {0},
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );
                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_shuffle(
                            ctx,
                            projected_part_shuffled,
                            lineitem_shuffled,
                            part_x_lineitem,
                            {0},
                            {0},
                            rapidsmpf::ndsh::KeepKeys::YES
                        )  // p_partkey, l_quantity, l_extendedprice
                    );
                } else {
                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_broadcast(
                            ctx,
                            part,
                            lineitem,
                            part_x_lineitem,
                            {0},
                            {0},
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::YES
                        )  // p_partkey, l_quantity, l_extendedprice
                    );
                }

                // Fanout the join result for two uses
                auto part_x_lineitem_for_avg = ctx->create_channel();
                auto part_x_lineitem_for_join = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::streaming::node::fanout(
                        ctx,
                        part_x_lineitem,
                        {part_x_lineitem_for_avg, part_x_lineitem_for_join},
                        rapidsmpf::streaming::node::FanoutPolicy::UNBOUNDED
                    )
                );

                // Compute average quantity grouped by p_partkey
                auto avg_quantity_chunks = ctx->create_channel();
                nodes.push_back(compute_avg_quantity(
                    ctx, part_x_lineitem_for_avg, avg_quantity_chunks
                ));

                // Concatenate average quantity results
                auto avg_quantity_concatenated = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx,
                        avg_quantity_chunks,
                        avg_quantity_concatenated,
                        rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                    )
                );

                // Final groupby for avg_quantity across all chunks
                auto avg_quantity_final = ctx->create_channel();
                nodes.push_back(groupby_avg_quantity(
                    ctx,
                    avg_quantity_concatenated,
                    avg_quantity_final,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                ));

                // Join part_x_lineitem with avg_quantity on p_partkey = key
                // avg_quantity_final is small (~199K rows), so broadcast it
                // part_x_lineitem is large, so keep it distributed
                auto final_join = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::inner_join_broadcast(
                        ctx,
                        avg_quantity_final,  // Small table - broadcast this
                        part_x_lineitem_for_join,  // Large table - probe with this
                        final_join,
                        {0},  // key from avg_quantity
                        {0},  // p_partkey from part_x_lineitem
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
                        rapidsmpf::ndsh::KeepKeys::YES
                    )  // key, avg_quantity, l_quantity, l_extendedprice
                );

                // Final aggregation (filter, sum, divide by 7) - processes chunks
                // incrementally
                auto aggregated = ctx->create_channel();
                nodes.push_back(final_aggregation(
                    ctx,
                    final_join,
                    aggregated,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                ));

                // Round result
                auto rounded = ctx->create_channel();
                nodes.push_back(round_result(ctx, aggregated, rounded));

                // Write output
                nodes.push_back(write_parquet(ctx, rounded, output_path));
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> pipeline = end - start;
            start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Q17 Iteration");
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

    RAPIDSMPF_MPI(MPI_Barrier(mpi_comm));
    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
