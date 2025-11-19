/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <rapidsmpf/allgather/allgather.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::allreduce {

/**
 * @brief Reduction operators supported by `AllReduce`.
 *
 * These closely mirror the reduction operators from MPI.
 */
enum class ReduceOp : std::uint8_t {
    SUM,  ///< Sum / addition
    PROD,  ///< Product / multiplication
    MIN,  ///< Minimum
    MAX,  ///< Maximum
};

/**
 * @brief Type-erased reduction kernel used by `AllReduce`.
 *
 * The kernel must implement an associative binary operation over the contents of
 * two `PackedData` objects and accumulate the result into @p accum.
 *
 * Implementations must:
 *  - Treat @p accum as the running partial result.
 *  - Combine @p incoming into @p accum in-place.
 *  - Leave @p incoming in a valid but unspecified state after the call.
 *
 * The kernel is responsible for interpreting `PackedData::metadata` and
 * `PackedData::data` consistently across all ranks.
 */
using ReduceKernel = std::function<void(PackedData& accum, PackedData&& incoming)>;

/**
 * @brief AllReduce collective.
 *
 * The current implementation is built using `allgather::AllGather` and performs
 * the reduction locally after allgather completes. Considering `R` is the number of
 * ranks, and `N` is the number of bytes of data, per rank this incurs `O(R * N)` bytes of
 * memory consumption and `O(R)` communication operations.
 *
 * Semantics:
 *  - Each rank may call `insert` any number of times with a local sequence number.
 *  - Conceptually, the *k*-th insertion on each rank participates in a single
 *    global reduction. That is, insertions are paired across ranks by their
 *    local insertion order, not by sequence number values.
 *  - Once all ranks call `insert_finished`, `wait_and_extract` returns one
 *    globally-reduced `PackedData` per local insertion on this rank.
 *
 * The actual reduction is implemented via a type-erased `ReduceKernel` that is
 * supplied at construction time. Helper factories such as
 * `detail::make_reduce_kernel` can be used to build element-wise
 * reductions over contiguous arrays in device memory.
 */
class AllReduce {
  public:
    /**
     * @brief Construct a new AllReduce operation.
     *
     * @param comm The communicator for communication.
     * @param progress_thread The progress thread used by the underlying AllGather.
     * @param op_id Unique operation identifier for this allreduce.
     * @param br Buffer resource for memory allocation.
     * @param statistics Statistics collection instance (disabled by default).
     * @param reduce_kernel Type-erased reduction kernel to use.
     * @param finished_callback Optional callback run once locally when the allreduce
     *        is finished and results are ready for extraction.
     *
     * @note This constructor internally creates an `allgather::AllGather` instance
     *       that uses the same communicator, progress thread, and buffer resource.
     */
    AllReduce(
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        OpID op_id,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
        ReduceKernel reduce_kernel = {},
        std::function<void(void)> finished_callback = nullptr
    );

    AllReduce(AllReduce const&) = delete;
    AllReduce& operator=(AllReduce const&) = delete;
    AllReduce(AllReduce&&) = delete;
    AllReduce& operator=(AllReduce&&) = delete;

    /**
     * @brief Destructor.
     *
     * @note This operation is logically collective. If an `AllReduce` is locally
     *       destructed before `wait_and_extract` is called, there is no guarantee
     *       that in-flight communication will be completed.
     */
    ~AllReduce();

    /**
     * @brief Insert packed data into the allreduce operation.
     *
     * @param sequence_number Local ordered sequence number of the data.
     * @param packed_data The data to contribute to the allreduce.
     *
     * The caller promises that:
     *  - `sequence_number`s are non-decreasing on each rank.
     *  - The *k*-th call to `insert` on each rank corresponds to the same logical
     *    reduction across all ranks (i.e., same element type and shape).
     */
    void insert(std::uint64_t sequence_number, PackedData&& packed_data);

    /**
     * @brief Mark that this rank has finished contributing data.
     */
    void insert_finished();

    /**
     * @brief Check if the allreduce operation has completed.
     *
     * @return True if all data and finish messages from all ranks have been
     *         received and locally reduced.
     */
    [[nodiscard]] bool finished() const noexcept;

    /**
     * @brief Wait for completion and extract all reduced data.
     *
     * Blocks until the allreduce operation completes and returns all locally
     * reduced results, ordered by local insertion order.
     *
     * @param timeout Optional maximum duration to wait. Negative values mean
     *        no timeout.
     *
     * @return A vector containing reduced packed data, one entry per local
     *         insertion on this rank.
     * @throws std::runtime_error If the timeout is reached.
     */
    [[nodiscard]] std::vector<PackedData> wait_and_extract(
        std::chrono::milliseconds timeout = std::chrono::milliseconds{-1}
    );

    /**
     * @brief Check if reduced results are ready for extraction.
     *
     * This returns true once the underlying allgather has completed and, if
     * `wait_and_extract` has not yet been called, indicates that calling it
     * would not block.
     *
     * @return True if the allreduce operation has completed and results are ready for
     *         extraction, false otherwise.
     */
    [[nodiscard]] bool is_ready() const noexcept;

  private:
    /// @brief Perform the reduction across all ranks for all gathered contributions.
    [[nodiscard]] std::vector<PackedData> reduce_all(std::vector<PackedData>&& gathered);

    std::shared_ptr<Communicator> comm_;  ///< Communicator
    std::shared_ptr<ProgressThread>
        progress_thread_;  ///< Progress thread (unused directly).
    BufferResource* br_;  ///< Buffer resource
    std::shared_ptr<Statistics> statistics_;  ///< Statistics collection instance
    ReduceKernel reduce_kernel_;  ///< Type-erased reduction kernel
    std::function<void(void)> finished_callback_;  ///< Optional finished callback

    allgather::AllGather gatherer_;  ///< Underlying allgather primitive

    std::atomic<std::uint32_t> nlocal_insertions_{0};  ///< Number of local inserts
    std::atomic<bool> reduced_computed_{
        false
    };  ///< Whether the reduction has been computed
    std::vector<PackedData> reduced_results_;  ///< Cached reduced results
};

namespace detail {
/**
 * @brief Create a device-based element-wise reduction kernel for a given (T, Op).
 *
 * This kernel expects both `PackedData::data` buffers to reside in device memory.
 * Implementations are provided in `device_kernels.cu` for a subset of (T, Op)
 * combinations.
 */
template <typename T, ReduceOp Op>
ReduceKernel make_reduce_kernel();
}  // namespace detail

}  // namespace rapidsmpf::allreduce
