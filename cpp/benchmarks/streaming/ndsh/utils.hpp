/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <any>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::ndsh {
namespace detail {

/**
 * @brief List all parquet files in a given path.
 *
 * @param root_path The path to look in.
 *
 * @return If `root_path` names a regular file that ends with `.parquet` then a singleton
 * vector of just that file. If `root_path` is a directory, then a vector containing all
 * regular files in that directory whose name ends with `.parquet`, in the order they are
 * listed.
 *
 * @throws std::runtime_error if the `root_path` doesn't name a regular file or a
 * directory. Or if it does name a regular file, but that file doesn't end in `.parquet`.
 */
[[nodiscard]] std::vector<std::string> list_parquet_files(std::string const root_path);

/**
 * @brief Get the path to a given table
 *
 * @param input_directory Input directory
 * @param table_name Name of table to find.
 *
 * @return Path to given table.
 */
[[nodiscard]] std::string get_table_path(
    std::string const& input_directory, std::string const& table_name
);

/**
 * @brief Get cudf data types for all columns from parquet metadata.
 *
 * Reads parquet metadata to determine the cudf data type for each column.
 *
 * @param input_directory Directory containing input parquet files
 * @param table_name Name of the table (e.g., "lineitem")
 * @return Map from column name to cudf data type
 */
[[nodiscard]] std::map<std::string, cudf::data_type> get_column_types(
    std::string const& input_directory, std::string const& table_name
);

}  // namespace detail

/**
 * @brief Create a date comparison filter expression.
 *
 * Creates a filter that compares a date column against a literal date value.
 * The operation will be equivalent to
 * "<column_name> <op> DATE '<year>-<month>-<day>'".
 *
 * @tparam timestamp_type The timestamp type to use for the filter scalar
 * (e.g., cudf::timestamp_D or cudf::timestamp_ms)
 * @param stream CUDA stream to use
 * @param date The date to compare against
 * @param column_name The name of the column to compare
 * @param op The comparison operator (e.g., LESS, LESS_EQUAL, GREATER)
 * @return Filter expression with proper lifetime management
 */
template <typename timestamp_type>
std::unique_ptr<streaming::Filter> make_date_filter(
    rmm::cuda_stream_view stream,
    cuda::std::chrono::year_month_day date,
    std::string const& column_name,
    cudf::ast::ast_operator op
) {
    auto owner = new std::vector<std::any>;
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
    owner->push_back(std::make_shared<cudf::ast::column_name_reference>(column_name));
    owner->push_back(
        std::make_shared<cudf::ast::operation>(
            op,
            *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(
                owner->at(2)
            ),
            *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(1))
        )
    );
    return std::make_unique<streaming::Filter>(
        stream,
        *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
        OwningWrapper(static_cast<void*>(owner), [](void* p) {
            delete static_cast<std::vector<std::any>*>(p);
        })
    );
}

/**
 * @brief Sink messages into a channel and discard them.
 *
 * @param ctx Streaming context
 * @param ch Channel to discard messages from.
 *
 * @return Coroutine representing the shutdown and discard of the channel.
 */
[[nodiscard]] streaming::Node sink_channel(
    std::shared_ptr<streaming::Context> ctx, std::shared_ptr<streaming::Channel> ch
);

/**
 * @brief Consume messages from a channel and discard them.
 *
 * @param ctx Streaming context
 * @param ch Channel to consume messages from.
 *
 * @note If the channel contains `TableChunk`s, moves them to device and prints small
 * amount of detail about them (row and column count).
 *
 * @return Coroutine representing consuming and discarding messages in channel.
 */
[[nodiscard]] streaming::Node consume_channel(
    std::shared_ptr<streaming::Context> ctx, std::shared_ptr<streaming::Channel> ch_in
);

///< @brief Communicator type to use
enum class CommType : std::uint8_t {
    SINGLE,  ///< Single process communicator
    MPI,  ///< MPI backed communicator
    UCXX,  ///< UCXX backed communicator
    MAX,  ///< Max value
};

///< @brief Configuration options for the query
struct ProgramOptions {
    int num_streaming_threads{1};  ///< Number of streaming threads to use
    int num_iterations{2};  ///< Number of iterations of query to run
    int num_streams{16};  ///< Number of streams in stream pool
    CommType comm_type{CommType::UCXX};  ///< Type of communicator to create
    std::optional<std::chrono::milliseconds>
        periodic_spill;  ///< Duration between background periodic spilling checks
    cudf::size_type num_rows_per_chunk{
        100'000'000
    };  ///< Number of rows to produce per chunk read
    std::optional<double> spill_device_limit{
        std::nullopt
    };  ///< Optional fractional spill limit
    bool no_pinned_host_memory{false};  ///< Disable pinned host memory?
    bool use_shuffle_join = false;  ///< Use shuffle join for "big" joins?
    std::string output_file;  ///< File to write output to
    std::string input_directory;  ///< Directory containing input files.
};

/**
 * @brief Parse commandline arguments
 *
 * @param argc Number of arguments
 * @param argv Arguments
 *
 * @return `ProgramOptions` struct with parsed arguments.
 */
ProgramOptions parse_arguments(int argc, char** argv);

/**
 * @brief Create a streaming execution context for a query.
 *
 * @param arguments Arguments to configure the context
 * @param mr Pointer to memory resource to use for all allocations
 * @warning The memory resource _must_ be kept alive until the final usage of the returned
 * Context is complete.
 *
 * @return Shared pointer to new streaming context.
 */
std::shared_ptr<streaming::Context> create_context(
    ProgramOptions& arguments, RmmResourceAdaptor* mr
);

/**
 * @brief Finalize MPI when going out of scope.
 */
struct FinalizeMPI {
    ~FinalizeMPI() noexcept {
        if (rapidsmpf::mpi::is_initialized()) {
            int flag;
            RAPIDSMPF_MPI(MPI_Finalized(&flag));
            if (!flag) {
                RAPIDSMPF_MPI(MPI_Finalize());
            }
        }
    }
};
}  // namespace rapidsmpf::ndsh
