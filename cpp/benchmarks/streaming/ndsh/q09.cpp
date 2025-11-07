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
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
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
                       .columns(
                           {"l_discount",
                            "l_extendedprice",
                            "l_orderkey",
                            "l_partkey",
                            "l_quantity",
                            "l_suppkey"}
                       )
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_nation(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "nation")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"n_name", "n_nationkey"})
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
        get_table_path(input_directory, "orders")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"o_orderdate", "o_orderkey"})
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
                       .columns({"p_partkey", "p_name"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_partsupp(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "partsupp")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"ps_partkey", "ps_suppkey", "ps_supplycost"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_supplier(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        get_table_path(input_directory, "supplier")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"s_nationkey", "s_suppkey"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

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
        auto p_name = table.column(1);
        auto target = cudf::make_string_scalar("green", chunk_stream, mr);
        auto mask = cudf::strings::contains(
            p_name, *static_cast<cudf::string_scalar*>(target.get()), chunk_stream, mr
        );
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                msg.sequence_number(),
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    cudf::apply_boolean_mask(
                        table.select({0}), mask->view(), chunk_stream, mr
                    ),
                    chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

[[maybe_unused]] rapidsmpf::streaming::Node select_columns(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    // n_name, ps_supplycost, l_discount, l_extendedprice, l_quantity,  o_orderdate

    // Select n_name, year_part_of(o_orderdate), amount = (extendedprice * (1
    // - discount)) - (ps_supplycost * l_quantity) group by n_name year agg
    // sum(amount).round(2) sort by n_name, o_year descending = true, false
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
        result.reserve(3);
        // n_name
        result.push_back(
            std::make_unique<cudf::column>(
                table.column(0), chunk_stream, ctx->br()->device_mr()
            )
        );
        result.push_back(
            cudf::datetime::extract_datetime_component(
                table.column(5),
                cudf::datetime::datetime_component::YEAR,
                chunk_stream,
                ctx->br()->device_mr()
            )
        );
        auto discount = table.column(2);
        auto extendedprice = table.column(3);
        auto supplycost = table.column(1);
        auto quantity = table.column(4);
        std::string udf =
            R"***(
static __device__ void calculate_amount(double *amount, double discount, double extprice, double supplycost, double quantity) {
    *amount = extprice * (1 - discount) - supplycost * quantity;
}
           )***";
        result.push_back(
            cudf::transform(
                {discount, extendedprice, supplycost, quantity},
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

[[maybe_unused]] rapidsmpf::streaming::Node chunkwise_groupby_agg(
    [[maybe_unused]] std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::vector<cudf::table> partial_results;
    std::uint64_t sequence = 0;
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        ctx->comm()->logger().print("Chunkwise groupby");
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        auto grouper = cudf::groupby::groupby(
            table.select({0, 1}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(2), std::move(aggs))
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
            table.select({0, 1}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        auto requests = std::vector<cudf::groupby::aggregation_request>();
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(2), std::move(aggs))
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
            if (ctx->comm()->rank() == 0) {
                // We will only actually bother to do this on rank zero.
                auto result_view = global_result->view();
                auto grouper = cudf::groupby::groupby(
                    result_view.select({0, 1}),
                    cudf::null_policy::EXCLUDE,
                    cudf::sorted::NO
                );
                auto requests = std::vector<cudf::groupby::aggregation_request>();
                std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
                aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
                requests.push_back(
                    cudf::groupby::aggregation_request(
                        result_view.column(2), std::move(aggs)
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
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto table = chunk.table_view();
    auto rounded = cudf::round(
        table.column(2),
        2,
        cudf::rounding_method::HALF_EVEN,
        chunk.stream(),
        ctx->br()->device_mr()
    );
    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                cudf::sort_by_key(
                    cudf::table_view({table.column(0), table.column(1), rounded->view()}),
                    table.select({0, 1}),
                    {cudf::order::ASCENDING, cudf::order::DESCENDING},
                    {cudf::null_order::BEFORE, cudf::null_order::BEFORE},
                    chunk.stream(),
                    ctx->br()->device_mr()
                ),
                chunk.stream()
            )
        )
    );
    co_await ch_out->drain(ctx->executor());
}

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
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("nation");
    metadata.column_metadata[1].set_name("o_year");
    metadata.column_metadata[2].set_name("sum_profit");
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
    rmm::mr::cuda_memory_resource base{};
    // rmm::mr::pool_memory_resource mr{&base, limit_size};
    auto stats_mr = rapidsmpf::RmmResourceAdaptor(&mr);
    rmm::device_async_resource_ref mr_ref(stats_mr);
    rmm::mr::set_current_device_resource(&stats_mr);
    rmm::mr::set_current_device_resource_ref(mr_ref);
    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
        memory_available{};
    memory_available[rapidsmpf::MemoryType::DEVICE] =
        rapidsmpf::LimitAvailableMemory{&stats_mr, static_cast<std::int64_t>(limit_size)};
    rapidsmpf::BufferResource br(stats_mr);  // , std::move(memory_available));
    auto envvars = rapidsmpf::config::get_environment_variables();
    envvars["num_streaming_threads"] = std::to_string(cmd_options.num_streaming_threads);
    auto options = rapidsmpf::config::Options(envvars);
    auto stats = std::make_shared<rapidsmpf::Statistics>(&stats_mr);
    {
        auto comm = rapidsmpf::ucxx::init_using_mpi(mpi_comm, options);
        // auto comm = std::make_shared<rapidsmpf::MPI>(mpi_comm, options);
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
                RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q9 pipeline");
                auto part = ctx->create_channel();
                auto filtered_part = ctx->create_channel();
                auto partsupp = ctx->create_channel();
                auto part_x_partsupp = ctx->create_channel();
                auto supplier = ctx->create_channel();
                auto lineitem = ctx->create_channel();
                auto supplier_x_part_x_partsupp = ctx->create_channel();
                auto supplier_x_part_x_partsupp_x_lineitem = ctx->create_channel();
                nodes.push_back(read_part(
                    ctx,
                    part,
                    /* num_tickets */ 2,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // p_partkey, p_name
                nodes.push_back(filter_part(ctx, part, filtered_part));  // p_partkey
                nodes.push_back(read_partsupp(
                    ctx,
                    partsupp,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // ps_partkey, ps_suppkey, ps_supplycost
                nodes.push_back(
                    // p_partkey x ps_partkey
                    rapidsmpf::ndsh::inner_join_broadcast(
                        ctx,
                        filtered_part,
                        partsupp,
                        part_x_partsupp,
                        {0},
                        {0},
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                    )  // p_partkey/ps_partkey, ps_suppkey, ps_supplycost
                );
                nodes.push_back(read_supplier(
                    ctx,
                    supplier,
                    /* num_tickets */ 2,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // s_nationkey, s_suppkey
                nodes.push_back(
                    // s_suppkey x ps_suppkey
                    rapidsmpf::ndsh::inner_join_broadcast(
                        ctx,
                        supplier,
                        part_x_partsupp,
                        supplier_x_part_x_partsupp,
                        {1},
                        {1},
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}

                    )  // s_nationkey, s_suppkey/ps_suppkey, p_partkey/ps_partkey,
                       // ps_supplycost
                );
                nodes.push_back(read_lineitem(
                    ctx,
                    lineitem,
                    /* num_tickets */ 4,
                    cmd_options.num_rows_per_chunk,
                    cmd_options.input_directory
                ));  // l_discount, l_extendedprice, l_orderkey, l_partkey, l_quantity,
                // l_suppkey
                nodes.push_back(
                    // [p_partkey, ps_suppkey] x [l_partkey, l_suppkey]
                    rapidsmpf::ndsh::inner_join_broadcast(
                        ctx,
                        supplier_x_part_x_partsupp,
                        lineitem,
                        supplier_x_part_x_partsupp_x_lineitem,
                        {2, 1},
                        {3, 5},
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
                        rapidsmpf::ndsh::KeepKeys::NO
                    )  // s_nationkey, ps_supplycost,
                       // l_discount, l_extendedprice, l_orderkey, l_quantity
                );
                auto nation = ctx->create_channel();
                auto orders = ctx->create_channel();
                nodes.push_back(
                    read_nation(
                        ctx,
                        nation,
                        /* num_tickets */ 1,
                        cmd_options.num_rows_per_chunk,
                        cmd_options.input_directory
                    )  // n_name, n_nationkey
                );
                nodes.push_back(
                    read_orders(
                        ctx,
                        orders,
                        /* num_tickets */ 4,
                        cmd_options.num_rows_per_chunk,
                        cmd_options.input_directory
                    )  // o_orderdate, o_orderkey
                );
                auto all_joined = ctx->create_channel();
                auto supplier_x_part_x_partsupp_x_lineitem_x_orders =
                    ctx->create_channel();
                if (cmd_options.use_shuffle_join) {
                    auto supplier_x_part_x_partsupp_x_lineitem_shuffled =
                        ctx->create_channel();
                    auto orders_shuffled = ctx->create_channel();
                    std::uint32_t num_partitions = 8;
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            supplier_x_part_x_partsupp_x_lineitem,
                            supplier_x_part_x_partsupp_x_lineitem_shuffled,
                            {4},
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );
                    nodes.push_back(
                        rapidsmpf::ndsh::shuffle(
                            ctx,
                            orders,
                            orders_shuffled,
                            {1},
                            num_partitions,
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            }
                        )
                    );
                    nodes.push_back(
                        // l_orderkey x o_orderkey
                        rapidsmpf::ndsh::inner_join_shuffle(
                            ctx,
                            supplier_x_part_x_partsupp_x_lineitem_shuffled,
                            orders_shuffled,
                            supplier_x_part_x_partsupp_x_lineitem_x_orders,
                            {4},
                            {1},
                            rapidsmpf::ndsh::KeepKeys::NO
                        )  // s_nationkey, ps_supplycost, l_discount, l_extendedprice,
                           // l_quantity, o_orderdate
                    );
                } else {
                    nodes.push_back(
                        // l_orderkey x o_orderkey
                        rapidsmpf::ndsh::inner_join_broadcast(
                            ctx,
                            supplier_x_part_x_partsupp_x_lineitem,
                            orders,
                            supplier_x_part_x_partsupp_x_lineitem_x_orders,
                            {4},
                            {1},
                            rapidsmpf::OpID{
                                static_cast<rapidsmpf::OpID>(10 * i + op_id++)
                            },
                            rapidsmpf::ndsh::KeepKeys::NO
                        )  // s_nationkey, ps_supplycost, l_discount, l_extendedprice,
                           // l_quantity, o_orderdate
                    );
                }
                nodes.push_back(
                    // n_nationkey x s_nationkey
                    rapidsmpf::ndsh::inner_join_broadcast(
                        ctx,
                        nation,
                        supplier_x_part_x_partsupp_x_lineitem_x_orders,
                        all_joined,
                        {1},
                        {0},
                        rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
                        rapidsmpf::ndsh::KeepKeys::NO
                    )  // n_name, ps_supplycost, l_discount, l_extendedprice,
                    // l_quantity, o_orderdate
                );
                auto groupby_input = ctx->create_channel();
                nodes.push_back(select_columns(ctx, all_joined, groupby_input));
                auto chunkwise_groupby_output = ctx->create_channel();
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
                nodes.push_back(final_groupby_agg(
                    ctx,
                    concatenated_groupby_output,
                    groupby_output,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}
                ));
                auto sorted_output = ctx->create_channel();
                nodes.push_back(sort_by(ctx, groupby_output, sorted_output));
                nodes.push_back(write_parquet(ctx, sorted_output, output_path));
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> pipeline = end - start;
            start = std::chrono::steady_clock::now();
            {
                RAPIDSMPF_NVTX_SCOPED_RANGE("Q9 Iteration");
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
