/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/progress_thread.hpp>

/**
 * @namespace rapidsmpf
 * @brief Multi-Processor interfaces.
 */
namespace rapidsmpf {

/**
 * @typedef Rank
 * @brief The rank of a node (e.g. the rank of a MPI process), or world size (total number
 * of ranks).
 *
 * @note Ranks are always consecutive integers from zero to the total number of ranks.
 */
using Rank = std::int32_t;

/**
 * @typedef OpID
 * @brief Operation ID defined by the user. This allows users to concurrently execute
 * multiple operations, and each operation will be identified by its OpID.
 *
 * @note Although typed as an `int32`, the number of distinct operations is limited to
 * `2^20`.
 */
using OpID = std::int32_t;

/**
 * @typedef StageID
 * @brief Identifier for a stage of a communication operation.
 *
 * @note Although typed as an `int32`, the number of distinct stages is limited to
 * `2^3`.
 */
using StageID = std::int32_t;

/**
 * @brief A tag used for identifying messages in a communication operation.
 *
 * The tag is a 32-bit integer, with the following layout (low bits to high bits
 * left-to-right)
 *
 * @code{}
 * bits   | 012       | 34567 01234567 0123456 | 7 01234567
 *        |           |                        |
 * value  | stage (3) | operation (20)         | empty (9)
 *        |           |                        |
 * @endcode{}
 *
 * The restriction of 23 used bits comes from empirical MPI implementation limits for
 * `MPI_TAG_UB`. For example, when using UCX as a transport layer, OpenMPI is restricted
 * to `2^23` distinct tags.
 *
 * All messages in rapidsmpf over the same `Communicator` are disambiguated by `Tag`s. A
 * message sent with a given `Tag` can only be matched by a receive with a matching `Tag`.
 *
 * In the same way, collective operations over a `Communicator` are disambiguated by an
 * `OpID`: two different collectives with different `OpID`s will not interfere.
 *
 * Due to implementation restrictions, we are limited in the number of distinct tags we
 * can use (the `Tag` constructor checks for overflow but not reuse). In particular, we
 * support at most `2^20` distinct `OpIDs`, corresponding to `2^20` distinct collectives
 * running simultaneously.
 */
class Tag {
  public:
    /**
     * @typedef StorageT
     * @brief The physical data type to store the tag
     */
    using StorageT = std::int32_t;

    /// @brief Number of bits for the stage ID
    static constexpr int stage_id_bits{3};

    /// @brief Mask for the stage ID
    static constexpr StorageT stage_id_mask{(1 << stage_id_bits) - 1};

    /// @brief Number of bits for the operation ID
    static constexpr int op_id_bits{20};

    /// @brief Mask for the operation ID
    static constexpr StorageT op_id_mask{
        ((1 << (op_id_bits + stage_id_bits)) - 1) ^ stage_id_mask
    };

    /**
     * @brief Constructs a tag
     *
     * @throws std::overflow_error If either the `op` or `stage` values are negative or
     * too large.
     *
     * @param op The operation ID
     * @param stage The stage ID
     */
    constexpr Tag(OpID const op, StageID const stage)
        : tag_{
              (static_cast<StorageT>(op) << stage_id_bits) | static_cast<StorageT>(stage)
          } {
        RAPIDSMPF_EXPECTS(
            stage >= 0 && stage < (1 << stage_id_bits),
            "Invalid stage value",
            std::overflow_error
        );
        RAPIDSMPF_EXPECTS(
            op >= 0 && op < (1 << op_id_bits), "Invalid OpID value", std::overflow_error
        );
    }

    /**
     * @brief Returns the max number of bits used for the tag
     * @return bit length
     */
    [[nodiscard]] static constexpr std::size_t bit_length() noexcept {
        return op_id_bits + stage_id_bits;
    }

    /**
     * @brief Returns the max value of the tag
     * @return max value
     */
    [[nodiscard]] static constexpr StorageT max_value() noexcept {
        return (1 << bit_length()) - 1;
    }

    /**
     * @brief Returns the int32 view of the tag
     * @return int32 view of the tag
     */
    constexpr operator StorageT() const noexcept {
        return tag_;
    }

    /**
     * @brief Extracts the operation ID from the tag
     * @return The operation ID
     */
    [[nodiscard]] constexpr OpID op() const noexcept {
        return (tag_ & op_id_mask) >> stage_id_bits;
    }

    /**
     * @brief Extracts the stage ID from the tag
     * @return The stage ID
     */
    [[nodiscard]] constexpr StageID stage() const noexcept {
        return tag_ & stage_id_mask;
    }

  private:
    StorageT const tag_;
};

/**
 * @brief Exception thrown by the communication operations of an aborted communicator.
 *
 * See `Communicator::abort()` and `Communicator::is_aborted()`.
 */
class communicator_aborted : public std::runtime_error {
  public:
    /**
     * @brief Construct the exception.
     *
     * @param origin The rank that aborted the communicator.
     * @param reason The reason given by `origin`.
     */
    communicator_aborted(Rank origin, std::string const& reason);

    /**
     * @brief The rank that aborted the communicator.
     *
     * @return The rank.
     */
    [[nodiscard]] Rank origin() const noexcept;

    /**
     * @brief The reason given by the rank that aborted the communicator.
     *
     * @return The reason.
     */
    [[nodiscard]] std::string const& reason() const noexcept;

  private:
    Rank origin_;
    std::string reason_;
};

/**
 * @brief Abstract base class for a communication mechanism between nodes.
 *
 * Provides an interface for sending and receiving messages between nodes, supporting
 * asynchronous operations, GPU data transfers, and custom logging. Implementations must
 * define the virtual methods to enable specific communication backends.
 *
 * The API of the `Communicator` is _not_ stream-ordered (since the concrete libraries we
 * use to implement communication patterns are not stream-ordered). A consequence of this
 * is that the user must ensure that any stream-ordered work on buffers is complete before
 * passing a buffer into `send` or `recv`. As a corollary, after completing a future, the
 * extracted `Buffer` is valid on its stream (it has no stream-ordered work queued up).
 *
 * Sends and receives are matched on `(rank, tag)` pairs. The concrete implementation
 * _must_ provide that there is no message overtaking.
 *
 * Destroying a future while its communication is in flight, e.g., because the operation
 * it belongs to failed, aborts the communicator: the communication is kept alive (see
 * `FutureGraveyard`), but it may still match later messages with the same tag, so later
 * communication throws `communicator_aborted`. Create a new communicator to communicate
 * again.
 *
 * Implementing a backend: to get this, backend futures hand their in-flight
 * communication over to `graveyard()` when destroyed, the backend calls
 * `throw_if_aborted()` before posting new communication, and releases the graveyard
 * from its destructor, see `FutureGraveyard::release()`.
 */
class Communicator {
  private:
    /// @brief The abort state of a communicator, opaque, see `is_aborted()`.
    class AbortState;

  public:
    /**
     * @brief Abstract base class for asynchronous operation within the communicator.
     *
     * Encapsulates the concept of an asynchronous operation, allowing users to query
     * or wait for completion.
     */
    class Future {
      public:
        Future() = default;
        /// @brief Destructor.
        virtual ~Future() noexcept = default;
        Future(Future&&) = default;  ///< Movable.
        /**
         * @brief Move assignment
         *
         * @returns Moved this.
         */
        Future& operator=(Future&&) = default;
        Future(Future const&) = delete;  ///< Not copyable.
        Future& operator=(Future const&) = delete;  ///< Not copy-assignable
    };

    /**
     * @brief Keeps the in-flight communication of destroyed futures alive.
     *
     * Destroying a future does not cancel nor wait for its communication, so the
     * communication library may still access its buffers. A backend future destroyed
     * before its completion was observed, e.g., because the operation it belongs to
     * failed, hands its request and buffers over to the graveyard of its communicator
     * instead (see `graveyard()`), which keeps them alive. This aborts the communicator
     * (see `Communicator::is_aborted()`): the abandoned communication may still match
     * later messages with the same tag, so the communicator must not be used anymore.
     *
     * When the communicator is destroyed, its backend releases the graveyard (see
     * `release()`): the cancellation of the buried communication is requested where
     * possible, and what completes before a single deadline is freed. The graveyard is
     * shared with the futures, so it may outlive the communicator: what is buried after
     * `release()` is leaked.
     *
     * Burying never calls into the communication library, it only takes a lock: a future
     * may be destroyed anywhere, e.g., in a callback of the communication library or
     * after MPI is finalized.
     */
    class FutureGraveyard {
      public:
        /**
         * @brief The in-flight communication of a destroyed future: its request and
         * buffers.
         */
        class Grave {
          public:
            Grave() = default;
            /// @brief Destructor, frees the buffers.
            virtual ~Grave() noexcept = default;
            Grave(Grave const&) = delete;  ///< Not copyable.
            Grave& operator=(Grave const&) = delete;  ///< Not copy-assignable.

            /**
             * @brief Check whether the communication has completed.
             *
             * Must not wait for the communication to complete. It may make a progress
             * call that returns once the pending events are processed, e.g.,
             * `MPI_Test()`.
             *
             * @return True if the communication completed and its buffers may be freed.
             */
            [[nodiscard]] virtual bool completed() noexcept = 0;

            /**
             * @brief Request the cancellation of the communication, where possible.
             *
             * Called by `release()`. Must not block: whether the cancellation completed
             * is observed through `completed()`. The default does nothing.
             */
            virtual void cancel() noexcept {}
        };

        /**
         * @brief Destructor, leaks the graves that were not released.
         */
        ~FutureGraveyard() noexcept;

        FutureGraveyard(FutureGraveyard const&) = delete;  ///< Not copyable.
        FutureGraveyard& operator=(FutureGraveyard const&) = delete;  ///< Not assignable.

        /**
         * @brief Keep the in-flight communication of a destroyed future alive.
         *
         * Aborts the communicator. Only takes a lock, never calls into the communication
         * library. Leaks `grave` if it cannot be kept, or once `release()` was called.
         *
         * @param grave The communication, or null if it was leaked already, which still
         * aborts the communicator.
         */
        void bury(std::unique_ptr<Grave> grave) noexcept;

        /**
         * @brief Check whether anything was buried and not released yet.
         *
         * @return True if no grave is held.
         */
        [[nodiscard]] bool empty() const noexcept;

        /**
         * @brief Release the graves, when the communicator is destroyed.
         *
         * If anything was buried, requests the cancellation of all the buried
         * communication, then waits for it to complete against a single deadline of about
         * one second, however much there is, and frees the buffers of what completed.
         * What still did not complete, e.g., a send to a dead peer, is leaked rather than
         * freed while it may still be accessed. Must be called while the communication
         * library is usable. Graves buried afterwards are leaked.
         *
         * @param cancel Called once if anything was buried, before `Grave::cancel()` of
         * every grave, e.g., to cancel all in-flight communication of the communication
         * library at once. May be empty.
         * @param progress Called repeatedly while waiting, e.g., to progress the
         * communication library. May be empty.
         */
        void release(
            std::function<void()> const& cancel = nullptr,
            std::function<void()> const& progress = nullptr
        ) noexcept;

      private:
        friend class Communicator;

        /**
         * @brief Construct the graveyard of a communicator.
         *
         * @param abort_state The abort state of the communicator.
         */
        explicit FutureGraveyard(std::shared_ptr<AbortState> abort_state);

        std::shared_ptr<AbortState> const abort_state_;
        mutable std::mutex mutex_;  ///< Protects `graves_` and `released_`.
        std::vector<std::unique_ptr<Grave>> graves_;
        bool released_{false};
    };

  protected:
    /**
     * @brief Construct a communicator, with its abort state and graveyard.
     */
    Communicator();

  public:
    virtual ~Communicator() noexcept = default;

    /**
     * @brief Retrieves the rank of the current node.
     * @return The rank of the node.
     */
    [[nodiscard]] virtual Rank rank() const = 0;

    /**
     * @brief Retrieves the total number of ranks.
     * @return The total number of ranks.
     */
    [[nodiscard]] virtual Rank nranks() const = 0;

    /**
     * @brief Sends a host message to a specific rank.
     *
     * This is used to send data that resides in host memory and is guaranteed
     * to be valid at the time of the call.
     *
     * Use `release_sync_host_data` to obtain the data buffer again once the
     * future is completed.
     *
     * @param msg Unique pointer to the message data (host memory).
     * @param rank The destination rank.
     * @param tag Message tag for identification.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     *
     * @throws std::invalid_argument If @p msg is `nullptr`.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> send(
        std::unique_ptr<std::vector<std::uint8_t>> msg, Rank rank, Tag tag
    ) = 0;

    /**
     * @brief Sends a message (device or host) to a specific rank. Use `release_data` to
     * obtain the data buffer again once the future is completed.
     *
     * @param msg Unique pointer to the message data (Buffer).
     * @param rank The destination rank.
     * @param tag Message tag for identification.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     *
     * @throws std::invalid_argument If @p msg is `nullptr`.
     * @throws std::logic_error If @p msg is not ready (see warning for more details).
     *
     * @warning The caller is responsible to ensure the underlying `Buffer` allocation
     * and data are already valid before calling, for example, when a CUDA allocation
     * and/or copy are done asynchronously. Specifically, the caller should ensure
     * `Buffer::is_ready()` returns true before calling this function. Providing a
     * non-ready buffer leads to an irrecoverable condition.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, Tag tag
    ) = 0;

    /**
     * @brief Receives a message from a specific rank to a buffer. Use `release_data` to
     * extract the data out of the buffer once the future is completed.
     *
     * @param rank The source rank.
     * @param tag Message tag for identification.
     * @param recv_buffer The receive buffer.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     *
     * @throws std::invalid_argument If @p recv_buffer is `nullptr`.
     * @throws std::logic_error If @p recv_buffer is not ready (see warning for more
     * details).
     *
     * @warning The caller is responsible to ensure the underlying `Buffer` allocation
     * is already valid before calling, for example, when a CUDA allocation
     * and/or copy are done asynchronously. Specifically, the caller should ensure
     * `Buffer::is_ready()` returns true before calling this function. Providing a
     * non-ready buffer leads to an irrecoverable condition.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> recv(
        Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
    ) = 0;

    /**
     * @brief Receives a message from a specific rank to an allocated (synchronized) host
     * buffer. Use `release_sync_host_data` to extract the data out of the buffer once the
     * future is completed.
     *
     * @param rank The source rank.
     * @param tag Message tag for identification.
     * @param synced_buffer The receive buffer.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     *
     * @throws std::invalid_argument If @p synced_buffer is `nullptr`.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> recv_sync_host_data(
        Rank rank, Tag tag, std::unique_ptr<std::vector<std::uint8_t>> synced_buffer
    ) = 0;

    /**
     * @brief Receives a message from any rank (blocking).
     *
     * @param tag Message tag for identification.
     * @return A pair containing the message data (host memory) and the rank of the
     * sender.
     * @note If no message is available this is indicated by returning
     * a `nullptr` in the first slot of the pair.
     */
    [[nodiscard]] virtual std::pair<std::unique_ptr<std::vector<std::uint8_t>>, Rank>
    recv_any(Tag tag) = 0;

    /**
     * @brief Receives a message from a specific rank (blocking).
     *
     * @param src The source rank from which to receive the message.
     * @param tag Message tag for identification.
     * @return A unique pointer to a vector containing the received message data (host
     * memory).
     *
     * @note If no message is available, this function returns a nullptr.
     */
    [[nodiscard]] virtual std::unique_ptr<std::vector<std::uint8_t>> recv_from(
        Rank src, Tag tag
    ) = 0;

    /**
     * @brief Tests for completion of multiple futures.
     *
     * @param[inout] future_vector Vector of Future objects. Completed
     * futures are erased from the vector.
     * @return Pair of completed futures and indices of input vector that were completed.
     */
    [[nodiscard]] virtual std::
        pair<std::vector<std::unique_ptr<Future>>, std::vector<std::size_t>>
        test_some(std::vector<std::unique_ptr<Future>>& future_vector) = 0;

    /**
     * @brief Tests for completion of multiple futures in a map.
     *
     * @param[inout] future_map Map of futures identified by keys. Completed futures are
     * erased from the map.
     * @return Map of the completed futures, identified by their keys.
     */
    [[nodiscard]] virtual std::unordered_map<std::size_t, std::unique_ptr<Future>>
    test_some(std::unordered_map<std::size_t, std::unique_ptr<Future>>& future_map) = 0;

    /**
     * @brief Test for completion of a single future.
     *
     * @param future Future to test
     * @return True if the future is completed. After `test` returns true, it is safe to
     * call `release_data()`.
     */
    [[nodiscard]] virtual bool test(std::unique_ptr<Communicator::Future>& future) = 0;

    /**
     * @brief Wait for completion of all futures and return their data buffers.
     *
     * @param futures Futures to wait for completion of, consumed.
     * @return A vector of the contained data buffers.
     *
     * @throws communicator_aborted If the communicator is aborted, also while waiting.
     * The futures are then left in @p futures.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Buffer>> wait_all(
        std::vector<std::unique_ptr<Communicator::Future>>&& futures
    ) = 0;

    /**
     * @brief Wait for a future to complete and return the data buffer.
     *
     * @param future The future to wait for completion of.
     * @return A unique pointer to the GPU data buffer (or `nullptr` if the future had no
     * data).
     *
     * @throws communicator_aborted If the communicator is aborted, also while waiting.
     * The future is then destroyed, see `FutureGraveyard`.
     */
    [[nodiscard]] virtual std::unique_ptr<Buffer> wait(
        std::unique_ptr<Future> future
    ) = 0;

    /**
     * @brief Retrieves data associated with a completed future.
     *
     * @param future The completed future.
     * @return A unique pointer to the data buffer.
     *
     * @throws std::runtime_error if the future has no data.
     */
    [[nodiscard]] std::unique_ptr<Buffer> virtual release_data(
        std::unique_ptr<Communicator::Future> future
    ) = 0;

    /**
     * @brief Retrieves synchronized host data associated with a completed future. When
     * the future is completed, the the host data is valid, and ready, but  not
     * stream-ordered.
     *
     * @param future The completed future.
     * @return A unique pointer to the synchronized host data.
     *
     * @throws std::runtime_error if the future has no data.
     */
    [[nodiscard]] std::
        unique_ptr<std::vector<std::uint8_t>> virtual release_sync_host_data(
            std::unique_ptr<Communicator::Future> future
        ) = 0;

    /**
     * @brief Retrieves the logger associated with this communicator.
     * @return Shared pointer to the logger.
     */
    [[nodiscard]] virtual std::shared_ptr<Logger> const& logger() = 0;

    /**
     * @brief Retrieves the progress thread associated with this communicator.
     * @return Shared pointer to the progress thread.
     */
    [[nodiscard]] virtual std::shared_ptr<ProgressThread> const&
    progress_thread() const = 0;

    /**
     * @brief Provides a string representation of the communicator.
     * @return A string describing the communicator.
     */
    [[nodiscard]] virtual std::string str() const = 0;

    /**
     * @brief Abort the communicator on all ranks.
     *
     * Fail-safe for unrecoverable errors. When an operation fails on one rank, for
     * example a collective that raised an exception, the other ranks may wait forever
     * for messages that will never arrive. Aborting marks the communicator as aborted
     * on this rank and notifies the other ranks, best effort, which mark it as aborted
     * too. They are also notified if this rank already aborted the communicator, e.g.,
     * by destroying in-flight communication (see `FutureGraveyard`), which is the usual
     * case after a failed operation, unless another rank aborted it first, which
     * notified them. From then on, the communication operations of the communicator throw
     * `communicator_aborted`. Operations driven by the progress thread, such as the
     * shuffler and the collectives, then fail and rethrow it to their waiters, so
     * every rank gets an exception it can handle, e.g., by unwinding and exiting.
     *
     * The other ranks are notified by:
     *  - MPI: a message on a reserved tag, received when polling for messages.
     *  - UCXX: a control message, received by the UCX worker.
     *  - Single: there are no other ranks.
     *
     * In-flight communication is not cancelled; an aborted communicator must not be
     * used for communication anymore, only destroyed, see `FutureGraveyard`.
     *
     * If `terminate` is true, the processes of all ranks are terminated instead,
     * without unwinding: `MPI_Abort()` with MPI, and every rank terminating upon
     * receiving the abort message with UCXX. The other ranks are not notified
     * cooperatively first, so that they don't start unwinding. This is a last resort,
     * e.g., when other ranks may not be making progress.
     *
     * RapidsMPF never calls this itself: errors are reported to the caller as
     * exceptions, and it is up to the application to decide whether an error is
     * recoverable or whether the communicator must be aborted.
     *
     * @param reason Human-readable reason, forwarded to the other ranks and included in
     * the `communicator_aborted` message, best effort: each rank only records the first
     * reason it receives. If the communicator was already aborted, e.g., by destroying
     * in-flight communication or by another rank, the exceptions on this rank carry that
     * earlier reason, and so may those on the other ranks, e.g., if an operation driven
     * by the progress thread notified them first. The reason is always printed to
     * `stderr`.
     * @param terminate Whether to terminate the processes of all ranks.
     */
    void abort(std::string const& reason, bool terminate = false) noexcept;

    /**
     * @brief Check whether the communicator has been aborted, by any rank.
     *
     * A communicator is aborted by `abort()`, on this or another rank, or when a future
     * is destroyed while its communication is in flight, see `FutureGraveyard`.
     * Communication through an aborted communicator throws `communicator_aborted`.
     *
     * @note With backends that poll for abort notifications, such as MPI, an abort by
     * another rank is only observed once polled, e.g., by `throw_if_aborted()` or by
     * communication.
     *
     * @return True if the communicator has been aborted.
     */
    [[nodiscard]] bool is_aborted() const noexcept;

    /**
     * @brief Throw `communicator_aborted` if the communicator has been aborted.
     *
     * Polls for abort notifications first, if the backend requires it. If this rank
     * aborted the communicator by destroying in-flight communication, the other ranks
     * are notified the first time. Communication operations call this, and so do the
     * operations driven by the progress thread on every iteration, so that they fail
     * even when they have no communication pending.
     *
     * @throws communicator_aborted If the communicator has been aborted.
     */
    void throw_if_aborted();

  protected:
    /**
     * @brief Get the graveyard of the communicator.
     *
     * Backend futures hand their in-flight communication over to it when destroyed,
     * see `FutureGraveyard`.
     *
     * @return The graveyard, shared with the futures.
     */
    [[nodiscard]] std::shared_ptr<FutureGraveyard> const& graveyard() const noexcept;

    /**
     * @brief Record an abort received from another rank.
     *
     * Only the first abort is recorded, see `abort()`.
     *
     * @param origin The rank that aborted the communicator.
     * @param reason The reason given by `origin`.
     */
    void mark_aborted(Rank origin, std::string reason) noexcept;

    /**
     * @brief Receive pending abort notifications from other ranks.
     *
     * Backends that must poll for abort notifications override this and record them
     * with `mark_aborted()`. The default does nothing.
     */
    virtual void poll_abort() {}

    /**
     * @brief Notify the other ranks that the communicator has been aborted.
     *
     * Best effort: failures to notify a rank are ignored. The default notifies nobody,
     * for backends that cannot: the abort is then only observed on this rank.
     *
     * @param reason The reason for aborting.
     * @param terminate Whether the processes of all ranks must be terminated.
     */
    virtual void notify_abort(std::string const& reason, bool terminate) noexcept;

  private:
    std::shared_ptr<AbortState> abort_state_;
    std::shared_ptr<FutureGraveyard> graveyard_;
    /// @brief Whether the other ranks were notified of the abort by this rank.
    std::atomic<bool> abort_notified_{false};
};

/// @brief Whether RapidsMPF was built with the UCXX Communicator.
#ifdef RAPIDSMPF_HAVE_UCXX
constexpr bool COMM_HAVE_UCXX = true;
#else
constexpr bool COMM_HAVE_UCXX = false;
#endif

/// @brief Whether RapidsMPF was built with the MPI Communicator.
#ifdef RAPIDSMPF_HAVE_MPI
constexpr bool COMM_HAVE_MPI = true;
#else
constexpr bool COMM_HAVE_MPI = false;
#endif

/**
 * @brief Get all ranks of a communicator except the calling rank.
 *
 * @param comm The communicator.
 * @return The ranks in increasing order, excluding `comm.rank()`.
 */
[[nodiscard]] std::vector<Rank> all_remote_ranks(Communicator const& comm);

/**
 * @brief Overloads the stream insertion operator for the Communicator class.
 *
 * This function allows a description of a Communicator to be written to an output
 * stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, Communicator const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf
