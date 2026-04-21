/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/coll/utils.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/progress_thread.hpp>

namespace rapidsmpf::coll {

/**
 * @brief Sparse all-to-all collective over explicit source and destination peer sets.
 *
 * Each rank may send zero or more messages to ranks listed in `dsts` and
 * receives zero or more messages from ranks listed in `srcs`. Sender order is defined by
 * the local order of calls to `insert(dst, ...)` for each destination rank.
 *
 * This object is logically collective over the communicator and identified by `op_id`.
 * Local extraction is only valid after `wait()` has completed.
 */
class SparseAlltoall {
  public:
    /**
     * @brief Construct a sparse all-to-all collective instance.
     *
     * @param comm Communicator for the collective.
     * @param op_id Collective operation identifier.
     * @param br Buffer resource used for allocations.
     * @param srcs Ranks this rank will receive from.
     * @param dsts Ranks this rank will send to.
     * @param finished_callback Optional callback invoked exactly once when the collective
     * is locally complete. The callback should be fast and non-blocking. Ideally it
     * should only be used to signal a thread to do the actual work of extraction. Note in
     * particular that the callback should not extract any data.
     *
     * @throws std::out_of_range If either `srcs` or `dsts` have invalid values. All
     * source and destination ranks must be in `[0, ..., comm->nranks())`, and not equal
     * to the current rank.
     * @throws std::invalid_argument If the rank lists are not unique.
     * @throws std::logic_error If the communicator or buffer resource pointers are null.
     *
     * @note It is safe to reuse the `op_id` as soon as `wait` has completed
     * locally or the `finished_callback` has been invoked.
     *
     * @note The caller promises that inserted buffers are stream-ordered with respect
     * to their own stream, and extracted buffers are likewise guaranteed to be stream-
     * ordered with respect to their own stream.
     *
     * @note Collectively the src and dst pairs of participating ranks must be consistent
     * (not checked for). That is if rank-A advertises that rank-B is in its dst set,
     * rank-B must advertise that rank-A is in its src set. If we ever need to relax this
     * restriction we could have each rank advertise its send set and bootstrap the
     * two-sided information using the non-blocking consensus algorithm of Hoefler,
     * Siebert, and Lumsdaine, ACM SIGPLAN (2010),
     * https://dl.acm.org/doi/10.1145/1837853.1693476.
     */
    SparseAlltoall(
        std::shared_ptr<Communicator> comm,
        OpID op_id,
        BufferResource* br,
        std::vector<Rank> srcs,
        std::vector<Rank> dsts,
        std::function<void()>&& finished_callback = nullptr
    );

    ~SparseAlltoall() noexcept;

    SparseAlltoall(SparseAlltoall const&) = delete;
    SparseAlltoall& operator=(SparseAlltoall const&) = delete;
    SparseAlltoall(SparseAlltoall&&) = delete;
    SparseAlltoall& operator=(SparseAlltoall&&) = delete;

    /**
     * @brief Gets the communicator associated with this SparseAlltoall.
     *
     * @return Shared pointer to communicator.
     */
    [[nodiscard]] std::shared_ptr<Communicator> const& comm() const noexcept;

    /**
     * @brief Insert data to send to a destination rank.
     *
     * The order the destination rank obtains the sent data is given by the insertion
     * order on the send side. If inserting concurrently to the same destination, the
     * caller must establish a total order of the insertions, otherwise the reconstruction
     * order on the receive side is unspecified.
     *
     * @note Concurrent insertion by multiple threads is supported.
     *
     * @note the caller must ensure that `insert_finished()` is called _after_ all
     * `insert()` calls have completed.
     *
     * @param dst Destination rank. Must be present in the constructor's `dsts`.
     * @param packed_data Packed payload and metadata to send.
     */
    void insert(Rank dst, PackedData&& packed_data);

    /**
     * @brief Indicate that no more data will be inserted for any destination.
     *
     * Must be called exactly once.
     *
     * @note If multiple threads are `insert()`ing, you must establish a happens-before
     * relationship between the completion of all `insert()`s and the final call to
     * `insert_finished()`.
     */
    void insert_finished();

    /**
     * @brief Wait for local completion.
     *
     * @param timeout Optional timeout. Negative values mean no timeout.
     *
     * @throws std::runtime_error If the timeout is reached.
     */
    void wait(std::chrono::milliseconds timeout = std::chrono::milliseconds{-1});

    /**
     * @brief Extract all received messages from a source rank.
     *
     * The returned vector is ordered by the sender's local insertion order.
     *
     * @param src Source rank. Must be present in the constructor's `srcs`.
     * @return All messages received from `src`.
     *
     * @note Concurrent extraction is supported, behaviour is undefined if two threads
     * attempt to extract data from the same source.
     *
     * @throws std::logic_error If extracting before the collective is complete.
     */
    [[nodiscard]] std::vector<PackedData> extract(Rank src);

  private:
    struct SourceState {
        std::uint64_t expected_count{0};
        std::uint64_t received_count{0};
        std::vector<std::unique_ptr<detail::Chunk>> chunks{};
        std::vector<std::unique_ptr<detail::Chunk>> incoming{};

        [[nodiscard]] bool ready() const noexcept {
            return expected_count > 0 && expected_count == received_count;
        }
    };

    /// @brief Send all ready messages.
    void send_ready_messages();
    /// @brief Post receives for any outstanding metadata messages.
    void receive_metadata_messages();
    /// @brief Post receives for expected data messages.
    void receive_data_messages();
    /// @brief Complete receives for data messages.
    void complete_data_messages();
    /// @brief @return true if all internal containers are empty.
    [[nodiscard]] bool containers_empty() const;
    /// @brief @return Progress the communication state and return the progress state.
    [[nodiscard]] ProgressThread::ProgressState event_loop();

    std::shared_ptr<Communicator> comm_;
    BufferResource* br_;
    std::vector<Rank> srcs_;
    std::vector<Rank> dsts_;
    std::unordered_map<Rank, std::atomic<std::uint64_t>> next_ordinal_per_dst_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    OpID op_id_;
    std::atomic<bool> locally_finished_{false};
    bool can_extract_{false};

    detail::PostBox outgoing_{};
    std::vector<std::unique_ptr<detail::Chunk>> receive_posted_;
    std::vector<std::unique_ptr<Communicator::Future>> receive_futures_;
    std::vector<std::unique_ptr<Communicator::Future>> fire_and_forget_;
    std::unordered_map<Rank, SourceState> source_states_;
    std::function<void()> finished_callback_;
    ProgressThread::FunctionID function_id_;
};

}  // namespace rapidsmpf::coll
