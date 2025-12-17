/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "rapidsmpf/communicator/mpi.hpp"
#include "rapidsmpf/streaming/core/channel.hpp"
#include "rapidsmpf/streaming/core/node.hpp"

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


}  // namespace detail

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

/**
 * @brief Ensure a `TableChunk` is on device.
 *
 * @param ctx Streaming context
 * @param chunk Chunk to move from device, is left in a moved-from state
 * @param allow_overbooking Whether reserving memory is allowed to overbook
 *
 * @return New `TableChunk` on device
 * @throws std::overflow_error if overbooking is not allowed and not enough memory is
 * available to reserve.
 */
[[nodiscard]] streaming::TableChunk to_device(
    std::shared_ptr<streaming::Context> ctx,
    streaming::TableChunk&& chunk,
    bool allow_overbooking = false
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
