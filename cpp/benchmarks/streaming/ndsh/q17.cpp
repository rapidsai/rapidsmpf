/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0

 # FOR SF1K DATASET
 shape: (1, 1)
┌────────────┐
│ avg_yearly │
│ ---        │
│ f64        │
╞════════════╡
│ 3.0691e8   │
└────────────┘
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
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
#include "utilities.hpp"

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
                       .columns({"p_partkey", "p_brand", "p_container"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
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

// Filter part for p_brand == "Brand#23" and p_container == "MED BOX"
// Keep all three columns for the join
rapidsmpf::streaming::Node filter_part(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
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
        auto table = chunk.table_view();

        // p_partkey (0), p_brand (1), p_container (2)
        auto p_brand = table.column(1);
        auto p_container = table.column(2);

        auto brand_target = cudf::make_string_scalar("Brand#23", chunk_stream, mr);
        auto container_target = cudf::make_string_scalar("MED BOX", chunk_stream, mr);

        // Check p_brand == "Brand#23"
        auto brand_mask = cudf::binary_operation(
            p_brand,
            *brand_target,
            cudf::binary_operator::EQUAL,
            cudf::data_type(cudf::type_id::BOOL8),
            chunk_stream,
            mr
        );

        // Check p_container == "MED BOX"
        auto container_mask = cudf::binary_operation(
            p_container,
            *container_target,
            cudf::binary_operator::EQUAL,
            cudf::data_type(cudf::type_id::BOOL8),
            chunk_stream,
            mr
        );

        // Combine with AND
        auto combined_mask = cudf::binary_operation(
            brand_mask->view(),
            container_mask->view(),
            cudf::binary_operator::LOGICAL_AND,
            cudf::data_type(cudf::type_id::BOOL8),
            chunk_stream,
            mr
        );

        // Apply filter and keep all three columns (p_partkey, p_brand, p_container)
        auto filtered =
            cudf::apply_boolean_mask(table, combined_mask->view(), chunk_stream, mr);

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                msg.sequence_number(),
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(filtered), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

// Compute average quantity per part and apply filter
// Input: p_partkey (0), p_brand (1), p_container (2), l_quantity (3), l_extendedprice (4)
rapidsmpf::streaming::Node compute_avg_and_filter(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
    co_await ctx->executor()->schedule();

    // Collect all chunks for aggregation
    std::vector<std::unique_ptr<cudf::table>> tables;
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

        // Create a copy of the table
        std::vector<std::unique_ptr<cudf::column>> columns;
        for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
            columns.push_back(
                std::make_unique<cudf::column>(table.column(i), chunk_stream, mr)
            );
        }
        tables.push_back(std::make_unique<cudf::table>(std::move(columns)));
    }

    if (tables.empty()) {
        co_return;
    }

    // Concatenate all tables
    std::vector<cudf::table_view> table_views;
    for (const auto& tbl : tables) {
        table_views.push_back(tbl->view());
    }
    auto stream = ctx->br()->stream_pool().get_stream();
    auto concatenated = cudf::concatenate(table_views, stream, mr);
    auto concat_view = concatenated->view();

    // Debug: Log the number of columns
    ctx->comm()->logger().print(
        "compute_avg_and_filter: received table with ",
        concat_view.num_columns(),
        " columns"
    );

    // Table structure depends on KeepKeys setting in the join:
    // With KeepKeys::YES, the structure could be:
    // - 5 columns: p_partkey (0), p_brand (1), p_container (2), l_quantity (3),
    // l_extendedprice (4)
    //   (one key column from the join)
    // - 6 columns: p_partkey (0), p_brand (1), p_container (2), l_partkey (3), l_quantity
    // (4), l_extendedprice (5)
    //   (both key columns kept)
    cudf::table_view working_view;
    if (concat_view.num_columns() == 5) {
        // One key kept: p_partkey (0), p_brand (1), p_container (2), l_quantity (3),
        // l_extendedprice (4)
        working_view = concat_view.select({0, 3, 4});
    } else if (concat_view.num_columns() == 6) {
        // Both keys kept: p_partkey (0), p_brand (1), p_container (2), l_partkey (3),
        // l_quantity (4), l_extendedprice (5)
        working_view = concat_view.select({0, 4, 5});
    } else {
        ctx->comm()->logger().print(
            "ERROR: Unexpected number of columns: ", concat_view.num_columns()
        );
        co_return;
    }

    // Group by p_partkey and compute average quantity
    auto grouper = cudf::groupby::groupby(
        working_view.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
    );

    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
    requests.push_back(
        cudf::groupby::aggregation_request(working_view.column(1), std::move(aggs))
    );

    auto [keys, results] = grouper.aggregate(requests, stream, mr);

    // Now we have:
    // keys: p_partkey
    // results[0]: avg(l_quantity)

    // Multiply avg by 0.2
    auto point_two_scalar = cudf::make_fixed_width_scalar<double>(0.2, stream, mr);
    auto avg_quantity_times_02 = cudf::binary_operation(
        results[0].results[0]->view(),
        *point_two_scalar,
        cudf::binary_operator::MUL,
        cudf::data_type(cudf::type_id::FLOAT64),
        stream,
        mr
    );

    // Create table with p_partkey and avg_quantity (0.2 * avg)
    std::vector<std::unique_ptr<cudf::column>> avg_table_cols;
    avg_table_cols.push_back(
        std::make_unique<cudf::column>(keys->get_column(0), stream, mr)
    );
    avg_table_cols.push_back(std::move(avg_quantity_times_02));
    auto avg_table = std::make_unique<cudf::table>(std::move(avg_table_cols));

    // Join back with original data using the new cudf::inner_join API
    // which returns indices instead of a joined table
    auto [left_indices, right_indices] = cudf::inner_join(
        working_view.select({0}),  // p_partkey from working_view
        avg_table->view().select({0}),  // p_partkey from avg_table
        cudf::null_equality::EQUAL,
        stream,
        mr
    );

    // Convert device_uvectors to column_views for gather
    auto left_indices_col = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32},
        left_indices->size(),
        left_indices->data(),
        nullptr,  // no null mask
        0  // no nulls
    );
    auto right_indices_col = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32},
        right_indices->size(),
        right_indices->data(),
        nullptr,  // no null mask
        0  // no nulls
    );

    // Gather the necessary columns using the indices
    // We need: l_quantity, l_extendedprice from working_view, and avg_quantity from
    // avg_table
    auto gathered_working = cudf::gather(
        working_view, left_indices_col, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr
    );
    auto gathered_avg = cudf::gather(
        avg_table->view().select({1}),
        right_indices_col,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr
    );  // Just avg_quantity

    // Now we have:
    // gathered_working: p_partkey (0), l_quantity (1), l_extendedprice (2)
    // gathered_avg: avg_quantity (0)
    auto joined_view_working = gathered_working->view();
    auto joined_view_avg = gathered_avg->view();

    // Filter where l_quantity < avg_quantity
    auto filter_mask = cudf::binary_operation(
        joined_view_working.column(1),  // l_quantity
        joined_view_avg.column(0),  // avg_quantity
        cudf::binary_operator::LESS,
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr
    );

    auto filtered = cudf::apply_boolean_mask(
        joined_view_working.select({2}),  // Keep only l_extendedprice
        filter_mask->view(),
        stream,
        mr
    );

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(filtered), stream
            )
        )
    );
    co_await ch_out->drain(ctx->executor());
}

// Final aggregation: sum(l_extendedprice) / 7.0
rapidsmpf::streaming::Node final_aggregation(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
    co_await ctx->executor()->schedule();

    // Collect the filtered data
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    }

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();

    // Sum l_extendedprice locally
    std::unique_ptr<cudf::scalar> local_sum;
    if (table.num_rows() > 0) {
        local_sum = cudf::reduce(
            table.column(0),
            *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type(cudf::type_id::FLOAT64),
            chunk_stream,
            mr
        );
    } else {
        local_sum = cudf::make_fixed_width_scalar<double>(0.0, chunk_stream, mr);
    }

    if (ctx->comm()->nranks() > 1) {
        // Gather results across all ranks
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};

        // Pack the scalar value as a table
        auto sum_col = cudf::make_column_from_scalar(*local_sum, 1, chunk_stream, mr);
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(std::move(sum_col));
        auto sum_table = std::make_unique<cudf::table>(std::move(cols));

        auto pack = cudf::pack(sum_table->view(), chunk_stream, mr);
        gatherer.insert(
            0,
            {rapidsmpf::PackedData(
                std::move(pack.metadata),
                ctx->br()->move(std::move(pack.gpu_data), chunk_stream)
            )}
        );
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

            // Sum all the values
            auto global_sum = cudf::reduce(
                global_result->view().column(0),
                *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                cudf::data_type(cudf::type_id::FLOAT64),
                chunk_stream,
                mr
            );

            // Divide by 7.0
            auto seven_scalar =
                cudf::make_fixed_width_scalar<double>(7.0, chunk_stream, mr);
            auto avg_yearly_val =
                static_cast<cudf::numeric_scalar<double>*>(global_sum.get())
                    ->value(chunk_stream);
            auto result_val = avg_yearly_val / 7.0;

            // Round to 2 decimal places using C++ math (cudf::round is deprecated and
            // round_decimal doesn't support floats)
            double rounded_val = std::round(result_val * 100.0) / 100.0;
            auto result_scalar =
                cudf::make_fixed_width_scalar<double>(rounded_val, chunk_stream, mr);
            auto rounded_col =
                cudf::make_column_from_scalar(*result_scalar, 1, chunk_stream, mr);

            std::vector<std::unique_ptr<cudf::column>> result_cols;
            result_cols.push_back(std::move(rounded_col));
            auto result_table = std::make_unique<cudf::table>(std::move(result_cols));

            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    0,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::move(result_table), chunk_stream
                    )
                )
            );
        }
    } else {
        // Single rank case
        auto sum_val = static_cast<cudf::numeric_scalar<double>*>(local_sum.get())
                           ->value(chunk_stream);
        auto result_val = sum_val / 7.0;

        // Create result as a column
        auto result_scalar =
            cudf::make_fixed_width_scalar<double>(result_val, chunk_stream, mr);
        auto result_col =
            cudf::make_column_from_scalar(*result_scalar, 1, chunk_stream, mr);

        // Round to 2 decimal places using C++ math (cudf::round is deprecated and
        // round_decimal doesn't support floats)
        double rounded_val = std::round(result_val * 100.0) / 100.0;
        auto result_scalar_rounded =
            cudf::make_fixed_width_scalar<double>(rounded_val, chunk_stream, mr);
        auto rounded_col =
            cudf::make_column_from_scalar(*result_scalar_rounded, 1, chunk_stream, mr);

        std::vector<std::unique_ptr<cudf::column>> result_cols;
        result_cols.push_back(std::move(rounded_col));
        auto result_table = std::make_unique<cudf::table>(std::move(result_cols));

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(result_table), chunk_stream
                )
            )
        );
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

    // Print the final result value
    auto table_view = chunk.table_view();
    if (table_view.num_rows() > 0 && table_view.num_columns() > 0) {
        auto result_col = table_view.column(0);
        // Get the scalar value from the column (first row)
        auto scalar =
            cudf::get_element(result_col, 0, chunk.stream(), ctx->br()->device_mr());
        if (scalar->is_valid(chunk.stream())) {
            auto double_scalar = static_cast<cudf::numeric_scalar<double>*>(scalar.get());
            double final_value = double_scalar->value(chunk.stream());

            ctx->comm()->logger().print(
                "========================================\n",
                "Q17 RESULT:\n",
                "  avg_yearly = ",
                final_value,
                "\n",
                "========================================"
            );
        }
    }

    auto sink = cudf::io::sink_info(output_path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("avg_yearly");
    builder = builder.metadata(metadata);
    auto options = builder.build();
    cudf::io::write_parquet(options, chunk.stream());
    ctx->comm()->logger().print("Wrote result to ", output_path);
}

}  // namespace

struct ProgramOptions {
    int num_streaming_threads = 1;
    cudf::size_type num_rows_per_chunk = 100'000'000;
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
    auto limit_size = rmm::percent_of_free_device_memory(80);
    rmm::mr::cuda_async_memory_resource mr{};
    auto stats_mr = rapidsmpf::RmmResourceAdaptor(&mr);
    rmm::device_async_resource_ref mr_ref(stats_mr);
    rmm::mr::set_current_device_resource(&stats_mr);
    rmm::mr::set_current_device_resource_ref(mr_ref);
    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
        memory_available{};
    memory_available[rapidsmpf::MemoryType::DEVICE] =
        rapidsmpf::LimitAvailableMemory{&stats_mr, static_cast<std::int64_t>(limit_size)};
    rapidsmpf::BufferResource br(stats_mr);
    auto envvars = rapidsmpf::config::get_environment_variables();
    envvars["num_streaming_threads"] = std::to_string(cmd_options.num_streaming_threads);
    auto options = rapidsmpf::config::Options(envvars);
    auto stats = std::make_shared<rapidsmpf::Statistics>(&stats_mr);
    {
        auto comm = rapidsmpf::ucxx::init_using_mpi(mpi_comm, options);
        auto progress =
            std::make_shared<rapidsmpf::ProgressThread>(comm->logger(), stats);
        auto ctx =
            std::make_shared<rapidsmpf::streaming::Context>(options, comm, &br, stats);
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

                // Following the physical plan:
                // 1. SCAN PARQUET part
                auto part = ctx->create_channel();
                nodes.push_back(read_part(
                    ctx,
                    part,
                    /* num_tickets */ 2,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));

                // 2. Apply filter on part (p_brand == "Brand#23" AND p_container == "MED
                // BOX")
                auto filtered_part = ctx->create_channel();
                nodes.push_back(filter_part(ctx, part, filtered_part));

                // 4. SCAN PARQUET lineitem
                auto lineitem = ctx->create_channel();
                nodes.push_back(read_lineitem(
                    ctx,
                    lineitem,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));

                // 6. JOIN: part x lineitem
                // Result: p_partkey, p_brand, p_container, l_quantity, l_extendedprice
                auto joined = ctx->create_channel();
                if (cmd_options.use_shuffle_join) {
                    // 3. SHUFFLE filtered part
                    auto part_shuffled = ctx->create_channel();
                    std::uint32_t num_partitions = 8;
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            filtered_part,
                            part_shuffled,
                            {0},  // p_partkey
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );

                    // 5. SHUFFLE lineitem
                    auto lineitem_shuffled = ctx->create_channel();
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            lineitem,
                            lineitem_shuffled,
                            {0},  // l_partkey
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );

                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_shuffle(
                            ctx,
                            part_shuffled,
                            lineitem_shuffled,
                            joined,
                            {0},  // p_partkey
                            {0},  // l_partkey
                            rapidsmpf::ndsh::KeepKeys::YES
                        )  // Result: p_partkey, p_brand, p_container, l_quantity,
                           // l_extendedprice
                    );
                } else {
                    nodes.push_back(
                        rapidsmpf::ndsh::inner_join_broadcast(
                            ctx,
                            filtered_part,
                            lineitem,
                            joined,
                            {0},  // p_partkey
                            {0},  // l_partkey
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::YES
                        )  // Result: p_partkey, p_brand, p_container, l_quantity,
                           // l_extendedprice
                    );
                };

                // 7. Concatenate the joined data for aggregation
                auto concatenated = ctx->create_channel();
                nodes.push_back(
                    rapidsmpf::ndsh::concatenate(
                        ctx, joined, concatenated, rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                    )
                );

                // 8. Compute average quantity and filter
                auto filtered = ctx->create_channel();
                nodes.push_back(compute_avg_and_filter(ctx, concatenated, filtered));

                // 9. Final aggregation (sum and divide by 7.0)
                auto result = ctx->create_channel();
                nodes.push_back(final_aggregation(
                    ctx,
                    filtered,
                    result,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                ));

                // 10. Write output
                nodes.push_back(write_parquet(ctx, result, output_path));
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

    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
