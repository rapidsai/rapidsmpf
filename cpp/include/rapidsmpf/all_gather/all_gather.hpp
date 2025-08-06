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

namespace rapidsmpf::all_gather {

/**
 * @brief All-gather service for cuDF tables.
 *
 * The `AllGather` class provides an interface for performing an all-gather operation
 * on cuDF tables, collecting data from all ranks and distributing the complete dataset
 * to all ranks.
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
     * @brief Check if all ranks have finished sending their data.
     *
     * @return True if all ranks have finished, otherwise False.
     */
    [[nodiscard]] bool finished() const;

    /**
     * @brief Wait for all ranks to finish and extract all gathered data.
     *
     * @param ordered If true, the data is returned in order of the ranks and insertion
     * order.
     * @param timeout Optional timeout (ms) to wait.
     * @return A vector of packed data containing all gathered data from all ranks.
     *
     * @note There are no guarantees on the order of the data across ranks.
     *
     * @throw std::runtime_error if the timeout is reached.
     */
    [[nodiscard]] std::vector<PackedData> wait_and_extract(
        bool ordered = false, std::optional<std::chrono::milliseconds> timeout = {}
    );

  private:
    Communicator const* comm_;
    std::unique_ptr<shuffler::Shuffler> shuffler_;
    rmm::cuda_stream_view stream_;
    BufferResource* br_;
};

}  // namespace rapidsmpf::all_gather
