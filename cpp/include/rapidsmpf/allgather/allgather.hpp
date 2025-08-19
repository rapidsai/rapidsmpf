/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::allgather {

namespace detail {
using ChunkID = ::rapidsmpf::shuffler::detail::ChunkID;
}

/**
 * @brief All-gather service for packed data.
 *
 * The `AllGather` class provides an interface for performing an all-gather operation
 * on packed data, collecting data from all ranks and distributing the complete dataset
 * to all ranks.
 *
 * @note Ordering:
 * - `wait_and_extract`: No ordering guarantees.
 * - `wait_and_extract_ordered`: Data is ordered by rank and insertion order.
 *
 * @note Any empty packed data will be ignored during insertion.
 */
class AllGather {
  public:
    /**
     * @brief Construct a new all-gather for a single all-gather operation.
     *
     * @param comm The communicator to use.
     * @param progress_thread The progress thread to use.
     * @param op_id The operation ID of the all-gather. This ID is unique for this
     * operation, and should not be reused until all nodes has called
     * `AllGather::shutdown()`.
     * @param stream The CUDA stream for memory operations.
     * @param br Buffer resource used to allocate temporary and the all-gather result.
     * @param statistics The statistics instance to use (disabled by default).
     */
    AllGather(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        OpID op_id,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /**
     * @brief Destructor, which calls shutdown.
     */
    ~AllGather();

    /**
     * @brief Shutdown the all-gather, blocking until all inflight communication is done.
     */
    void shutdown();

    /**
     * @brief Insert data into the all-gather.
     *
     * @param data The data to insert.
     */
    void insert(PackedData&& data);

    /**
     * @brief Insert a finish mark indicating no more data will be inserted from this
     * rank.
     */
    void insert_finished();

    /**
     * @brief Check if data has been extracted.
     *
     * @return True if data has been extracted, otherwise False.
     */
    [[nodiscard]] bool finished() const;

    /**
     * @brief Wait for all ranks to finish and extract all gathered data.

     * @param timeout Optional timeout (ms) to wait.
     * @return A vector of non-empty packed data from all ranks.
     *
     * @note There are no guarantees on the order of the data across ranks.
     *
     * @throw std::runtime_error if the timeout is reached or if the data insertion
     * has not been finished yet.
     */
    [[nodiscard]] std::vector<PackedData> wait_and_extract(
        std::optional<std::chrono::milliseconds> timeout = {}
    );

    /**
     * @brief Wait for all ranks to finish and extract all gathered data.
     *
     * @param timeout Optional timeout (ms) to wait.
     * @return A vector of non-empty packed data ordered by rank and insertion order.
     *
     * @throw std::runtime_error if the timeout is reached or if the data insertion
     * has not been finished yet.
     */
    [[nodiscard]] std::vector<PackedData> wait_and_extract_ordered(
        std::optional<std::chrono::milliseconds> timeout = {}
    );

  private:
    Communicator* comm_;
    std::unique_ptr<shuffler::Shuffler> shuffler_;
    rmm::cuda_stream_view stream_;
    BufferResource* br_;

    std::atomic<bool> insert_finished_{false};
};

}  // namespace rapidsmpf::allgather
