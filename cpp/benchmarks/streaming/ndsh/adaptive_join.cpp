/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

#include <cuda_runtime_api.h>
#include <getopt.h>

#include <cudf/context.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>

#include "join.hpp"
#include "utils.hpp"

#include <coro/when_all.hpp>

namespace {

rapidsmpf::streaming::Node read_parquet(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::optional<std::vector<std::string>> columns,
    std::string const& input_directory,
    std::string const& input_file
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, input_file)
    );
    auto options =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info(files)).build();
    if (columns.has_value()) {
        options.set_columns(*columns);
    }
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

[[maybe_unused]] std::size_t estimate_read_parquet_messages(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::string const& input_directory,
    std::string const& input_file,
    cudf::size_type num_rows_per_chunk
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, input_file)
    );
    if (files.empty()) {
        return 0;
    }

    // Assumption: this file-to-rank mapping matches read_parquet in the streaming node.
    auto const rank = static_cast<std::size_t>(ctx->comm()->rank());
    auto const size = static_cast<std::size_t>(ctx->comm()->nranks());
    auto const base = files.size() / size;
    auto const extra = files.size() % size;
    auto const files_per_rank = base + (rank < extra ? 1 : 0);
    auto const file_offset = rank * base + std::min(rank, extra);
    if (files_per_rank == 0) {
        return 0;
    }

    std::size_t total_rows = 0;
    for (std::size_t i = 0; i < files_per_rank; ++i) {
        auto const& file = files[file_offset + i];
        total_rows += static_cast<std::size_t>(
            cudf::io::read_parquet_metadata(cudf::io::source_info(file)).num_rows()
        );
    }
    if (total_rows == 0 || num_rows_per_chunk <= 0) {
        return 0;
    }

    // Assumption: chunk sizes are close to num_rows_per_chunk and filters are absent.
    auto const chunk_rows = static_cast<std::size_t>(num_rows_per_chunk);
    return (total_rows + chunk_rows - 1) / chunk_rows;
}

[[maybe_unused]] rapidsmpf::streaming::Node advertise_message_count(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_meta,
    std::size_t estimate
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_meta};
    co_await ctx->executor()->schedule();
    auto payload = std::make_unique<std::size_t>(estimate);
    co_await ch_meta->send(rapidsmpf::streaming::Message{0, std::move(payload), {}, {}});
    co_await ch_meta->drain(ctx->executor());
    ctx->comm()->logger().print("Exiting message count");
}

[[maybe_unused]] rapidsmpf::streaming::Node consume_channel_parallel(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::size_t
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    std::size_t estimated_total_bytes{0};
    co_await ctx->executor()->schedule();
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        if (msg.holds<rapidsmpf::streaming::TableChunk>()) {
            auto chunk = rapidsmpf::ndsh::to_device(
                ctx, msg.release<rapidsmpf::streaming::TableChunk>()
            );
            ctx->comm()->logger().print(
                "Consumed chunk ",
                msg.sequence_number(),
                " with ",
                chunk.table_view().num_rows(),
                " rows and ",
                chunk.table_view().num_columns(),
                " columns"
            );
            estimated_total_bytes += chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE);
        }
    }
    ctx->comm()->logger().print(
        "Table was around ", rmm::detail::format_bytes(estimated_total_bytes)
    );
}

///< @brief Configuration options for the benchmark
struct ProgramOptions {
    int num_streaming_threads{1};  ///< Number of streaming threads to use
    int num_iterations{2};  ///< Number of iterations of query to run
    int num_streams{16};  ///< Number of streams in stream pool
    rapidsmpf::ndsh::CommType comm_type{
        rapidsmpf::ndsh::CommType::UCXX
    };  ///< Type of communicator to create
    cudf::size_type num_rows_per_chunk{
        100'000'000
    };  ///< Number of rows to produce per chunk read
    std::size_t num_producers{
        1
    };  ///< Number of simultaneous read_parquet chunk producers.
    std::size_t num_consumers{1};  ///< Number of simultaneous chunk consumers.
    std::string input_directory;  ///< Directory containing input files.
    std::string left_input_file;  ///< Basename of left input file to read.
    std::string right_input_file;  ///< Basename of right input file to read.
    std::optional<std::vector<std::string>> left_columns{
        std::nullopt
    };  ///< Columns to read (left input).
    std::optional<std::vector<std::string>> right_columns{
        std::nullopt
    };  ///< Columns to read (right input).
};

ProgramOptions parse_arguments(int argc, char** argv) {
    ProgramOptions options;

    static constexpr std::
        array<std::string_view, static_cast<std::size_t>(rapidsmpf::ndsh::CommType::MAX)>
            comm_names{"single", "mpi", "ucxx"};

    auto print_usage = [&argv, &options]() {
        std::cerr
            << "Usage: " << argv[0] << " [options]\n"
            << "Options:\n"
            << "  --num-streaming-threads <n>  Number of streaming threads (default: "
            << options.num_streaming_threads << ")\n"
            << "  --num-iterations <n>         Number of iterations (default: "
            << options.num_iterations << ")\n"
            << "  --num-streams <n>            Number of streams in stream pool "
               "(default: "
            << options.num_streams << ")\n"
            << "  --num-rows-per-chunk <n>     Number of rows per chunk (default: "
            << options.num_rows_per_chunk << ")\n"
            << "  --num-producers <n>          Number of concurrent read_parquet "
               "producers (default: "
            << options.num_producers << ")\n"
            << "  --num-consumers <n>          Number of concurrent consumers (default: "
            << options.num_consumers << ")\n"
            << "  --comm-type <type>           Communicator type: single, mpi, ucxx "
               "(default: "
            << comm_names[static_cast<std::size_t>(options.comm_type)] << ")\n"
            << "  --input-directory <path>     Input directory path (required)\n"
            << "  --left-input-file <file>     Left input file basename relative to "
               "input "
               "directory (required)\n"
            << "  --right-input-file <file>    Right input file basename relative to "
               "input directory (required)\n"
            << "  --left-columns <a,b,c>       Comma-separated column names to read "
               "(optional, default all columns)\n"
            << "  --right-columns <a,b,c>      Comma-separated column names to read "
               "(optional, default all columns)\n"
            << "  --help                       Show this help message\n";
    };

    // NOLINTBEGIN(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)
    static struct option long_options[] = {
        {"num-streaming-threads", required_argument, nullptr, 1},
        {"num-rows-per-chunk", required_argument, nullptr, 2},
        {"num-producers", required_argument, nullptr, 3},
        {"num-consumers", required_argument, nullptr, 4},
        {"input-directory", required_argument, nullptr, 5},
        {"left-input-file", required_argument, nullptr, 6},
        {"right-input-file", required_argument, nullptr, 12},
        {"help", no_argument, nullptr, 7},
        {"num-iterations", required_argument, nullptr, 8},
        {"num-streams", required_argument, nullptr, 9},
        {"comm-type", required_argument, nullptr, 10},
        {"left-columns", required_argument, nullptr, 11},
        {"right-columns", required_argument, nullptr, 13},
        {nullptr, 0, nullptr, 0}
    };
    // NOLINTEND(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-use-designated-initializers)

    int opt;
    int option_index = 0;

    bool saw_input_directory = false;
    bool saw_left_input_file = false;
    bool saw_right_input_file = false;

    auto parse_i64 = [](char const* s, char const* opt_name) -> long long {
        if (s == nullptr || *s == '\0') {
            std::cerr << "Error: " << opt_name << " requires a value\n";
            std::exit(1);
        }
        errno = 0;
        char* end = nullptr;
        auto const v = std::strtoll(s, &end, 10);
        if (errno != 0 || end == s || *end != '\0') {
            std::cerr << "Error: invalid integer for " << opt_name << ": '" << s << "'\n";
            std::exit(1);
        }
        return v;
    };

    auto parse_u64 = [](char const* s, char const* opt_name) -> unsigned long long {
        if (s == nullptr || *s == '\0') {
            std::cerr << "Error: " << opt_name << " requires a value\n";
            std::exit(1);
        }
        errno = 0;
        char* end = nullptr;
        auto const v = std::strtoull(s, &end, 10);
        if (errno != 0 || end == s || *end != '\0') {
            std::cerr << "Error: invalid non-negative integer for " << opt_name << ": '"
                      << s << "'\n";
            std::exit(1);
        }
        return v;
    };

    auto require_positive_i32 = [&](char const* s, char const* opt_name) -> int {
        auto const v = parse_i64(s, opt_name);
        if (v <= 0 || v > std::numeric_limits<int>::max()) {
            std::cerr << "Error: " << opt_name << " must be in [1, "
                      << std::numeric_limits<int>::max() << "], got '" << s << "'\n";
            std::exit(1);
        }
        return static_cast<int>(v);
    };

    auto require_positive_size_t = [&](char const* s,
                                       char const* opt_name) -> std::size_t {
        auto const v = parse_u64(s, opt_name);
        if (v == 0 || v > std::numeric_limits<std::size_t>::max()) {
            std::cerr << "Error: " << opt_name << " must be in [1, "
                      << std::numeric_limits<std::size_t>::max() << "], got '" << s
                      << "'\n";
            std::exit(1);
        }
        return static_cast<std::size_t>(v);
    };

    auto parse_columns = [](char const* s) -> std::optional<std::vector<std::string>> {
        if (s == nullptr) {
            return std::nullopt;
        }
        std::string str{s};
        if (str.empty()) {
            return std::nullopt;
        }
        std::vector<std::string> cols;
        std::size_t start = 0;
        while (start <= str.size()) {
            auto const comma = str.find(',', start);
            auto const end = (comma == std::string::npos) ? str.size() : comma;
            auto const token = str.substr(start, end - start);
            if (token.empty()) {
                std::cerr << "Error: --columns contains an empty column name\n";
                std::exit(1);
            }
            cols.push_back(token);
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return cols;
    };

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
        case 1:  // --num-streaming-threads
            options.num_streaming_threads =
                require_positive_i32(optarg, "--num-streaming-threads");
            break;
        case 2:  // --num-rows-per-chunk
            options.num_rows_per_chunk =
                require_positive_i32(optarg, "--num-rows-per-chunk");
            break;
        case 3:  // --num-producers
            options.num_producers = require_positive_size_t(optarg, "--num-producers");
            break;
        case 4:  // --num-consumers
            options.num_consumers = require_positive_size_t(optarg, "--num-consumers");
            break;
        case 5:  // --input-directory
            if (optarg == nullptr || *optarg == '\0') {
                std::cerr << "Error: --input-directory requires a non-empty value\n";
                std::exit(1);
            }
            options.input_directory = optarg;
            saw_input_directory = true;
            break;
        case 6:  // --left-input-file
            if (optarg == nullptr || *optarg == '\0') {
                std::cerr << "Error: --left-input-file requires a non-empty value\n";
                std::exit(1);
            }
            options.left_input_file = optarg;
            saw_left_input_file = true;
            break;
        case 12:  // --right-input-file
            if (optarg == nullptr || *optarg == '\0') {
                std::cerr << "Error: --right-input-file requires a non-empty value\n";
                std::exit(1);
            }
            options.right_input_file = optarg;
            saw_right_input_file = true;
            break;
        case 7:  // --help
            print_usage();
            std::exit(0);
        case 8:  // --num-iterations
            options.num_iterations = require_positive_i32(optarg, "--num-iterations");
            break;
        case 9:  // --num-streams
            options.num_streams = require_positive_i32(optarg, "--num-streams");
            break;
        case 10:
            {  // --comm-type
                if (optarg == nullptr || *optarg == '\0') {
                    std::cerr << "Error: --comm-type requires a value\n";
                    std::exit(1);
                }
                std::string_view const s{optarg};
                auto parsed = std::optional<rapidsmpf::ndsh::CommType>{};
                for (std::size_t i = 0; i < comm_names.size(); ++i) {
                    if (s == comm_names[i]) {
                        parsed = static_cast<rapidsmpf::ndsh::CommType>(i);
                        break;
                    }
                }
                if (!parsed.has_value()) {
                    std::cerr << "Error: invalid --comm-type '" << s
                              << "' (expected: single, mpi, ucxx)\n";
                    std::exit(1);
                }
                options.comm_type = *parsed;
                break;
            }
        case 11:  // --left-columns
            options.left_columns = parse_columns(optarg);
            break;
        case 13:  // --right-columns
            options.right_columns = parse_columns(optarg);
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
    if (!saw_input_directory || !saw_left_input_file || !saw_right_input_file) {
        if (!saw_input_directory) {
            std::cerr << "Error: --input-directory is required\n";
        }
        if (!saw_left_input_file) {
            std::cerr << "Error: --left-input-file is required\n";
        }
        if (!saw_right_input_file) {
            std::cerr << "Error: --right-input-file is required\n";
        }
        std::cerr << std::endl;
        print_usage();
        std::exit(1);
    }

    return options;
}

}  // namespace

/**
 * @brief Run a simple benchmark reading a table from parquet files.
 */
int main(int argc, char** argv) {
    rapidsmpf::ndsh::FinalizeMPI finalize{};
    cudaFree(nullptr);
    // work around https://github.com/rapidsai/cudf/issues/20849
    cudf::initialize();
    auto mr = rmm::mr::cuda_async_memory_resource{};
    auto stats_wrapper = rapidsmpf::RmmResourceAdaptor(&mr);
    auto arguments = parse_arguments(argc, argv);
    rapidsmpf::ndsh::ProgramOptions ctx_arguments{
        .num_streaming_threads = arguments.num_streaming_threads,
        .num_iterations = arguments.num_iterations,
        .num_streams = arguments.num_streams,
        .comm_type = arguments.comm_type,
        .num_rows_per_chunk = arguments.num_rows_per_chunk,
        .output_file = "",
        .input_directory = arguments.input_directory
    };

    auto ctx = rapidsmpf::ndsh::create_context(ctx_arguments, &stats_wrapper);
    std::vector<double> timings;
    for (int i = 0; i < arguments.num_iterations; i++) {
        std::vector<rapidsmpf::streaming::Node> nodes;
        int op_id = 0;
        auto start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing read_parquet pipeline");

            // Input data channels
            auto left_out = ctx->create_channel();
            auto right_out = ctx->create_channel();
            auto left_meta = ctx->create_channel();
            auto right_meta = ctx->create_channel();
            auto left_estimate = estimate_read_parquet_messages(
                ctx,
                arguments.input_directory,
                arguments.left_input_file,
                arguments.num_rows_per_chunk
            );
            auto right_estimate = estimate_read_parquet_messages(
                ctx,
                arguments.input_directory,
                arguments.right_input_file,
                arguments.num_rows_per_chunk
            );
            nodes.push_back(read_parquet(
                ctx,
                left_out,
                arguments.num_producers,
                arguments.num_rows_per_chunk,
                arguments.left_columns,
                arguments.input_directory,
                arguments.left_input_file
            ));
            nodes.push_back(read_parquet(
                ctx,
                right_out,
                arguments.num_producers,
                arguments.num_rows_per_chunk,
                arguments.right_columns,
                arguments.input_directory,
                arguments.right_input_file
            ));
            nodes.push_back(advertise_message_count(ctx, left_meta, left_estimate));
            nodes.push_back(advertise_message_count(ctx, right_meta, right_estimate));
            auto joined = ctx->create_channel();
            auto const size_tag = static_cast<rapidsmpf::OpID>(10 * i) + op_id++;
            auto const left_shuffle_tag = static_cast<rapidsmpf::OpID>(10 * i) + op_id++;
            auto const right_shuffle_tag = static_cast<rapidsmpf::OpID>(10 * i) + op_id++;
            nodes.push_back(
                rapidsmpf::ndsh::adaptive_inner_join(
                    ctx,
                    left_out,
                    right_out,
                    left_meta,
                    right_meta,
                    joined,
                    {0},
                    {0},
                    size_tag,
                    left_shuffle_tag,
                    right_shuffle_tag
                )
            );
            nodes.push_back(
                consume_channel_parallel(ctx, joined, arguments.num_consumers)
            );
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> pipeline = end - start;
        start = std::chrono::steady_clock::now();
        {
            RAPIDSMPF_NVTX_SCOPED_RANGE("read_parquet iteration");
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
