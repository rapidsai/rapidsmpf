/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

/**
 * @brief The collective responsible for a completed receive transfer.
 */
enum class CollectiveKind : std::uint8_t {
    ALLGATHER,
    SPARSE_ALLTOALL,
    SHUFFLER,
    ALLREDUCE,
};

/**
 * @brief A completed, data-bearing collective receive.
 */
struct TransferEvent {
    std::int32_t op_id{};  ///< Collective operation identifier.
    CollectiveKind collective_kind{};  ///< Collective implementation.
    std::int32_t source_rank{};  ///< Immediate wire source.
    std::int32_t destination_rank{};  ///< Rank that completed the receive.
    std::uint64_t message_id{};  ///< Stable chunk identifier or receive sequence.
    std::uint64_t metadata_bytes{};  ///< Logical application metadata bytes.
    std::uint64_t payload_bytes{};  ///< Payload bytes.
    MemoryType destination_memory_type{};  ///< Memory tier of the received payload.
    std::int64_t completion_timestamp_ns{};  ///< Steady-clock completion timestamp.
};

/**
 * @brief A progress thread that can execute arbitrary functions.
 *
 * Execute each of the registered arbitrary functions in a separate thread. The
 * functions are executed in the order they were registered, and a newly registered
 * function will only execute for the first time in the next iteration of the
 * progress thread.
 */
class ProgressThread {
  public:
    /**
     * @brief The progress state of a function, can be either `InProgress` or `Done`.
     */
    enum ProgressState : bool {
        InProgress,
        Done,
    };

    /**
     * @typedef FunctionIndex
     * @brief The sequential index of a function within a ProgressThread.
     */
    using FunctionIndex = std::uint64_t;

    /**
     * @typedef ProgressThreadAddress
     * @brief The address of a ProgressThread instance.
     */
    using ProgressThreadAddress = std::uintptr_t;

    /**
     * @brief The unique ID of a function registered with `ProgressThread`.
     * Composed of the ProgressThread address and a sequential function index.
     */
    struct FunctionID {
        ProgressThreadAddress thread_address{
            ProgressThreadAddress(0)
        };  ///< The address of the ProgressThread instance
        FunctionIndex function_index{0};  ///< The sequential index of the function

        /**
         * @brief Construct a FunctionID with an invalid address.
         *
         * For a valid object the constructor that takes `thread_addr` and `index`
         * must be used.
         *
         * @note This is the default constructor.
         */
        FunctionID() = default;

        /**
         * @brief Construct a new FunctionID
         *
         * @param thread_addr The address of the ProgressThread instance
         * @param index The sequential index of the function
         */
        constexpr FunctionID(ProgressThreadAddress thread_addr, FunctionIndex index)
            : thread_address(thread_addr), function_index(index) {}

        /**
         * @brief Check if the FunctionID is valid.
         *
         * @return True if the FunctionID is valid, false otherwise.
         */
        [[nodiscard]] constexpr bool is_valid() const {
            return thread_address != ProgressThreadAddress(0);
        }
    };

    /**
     * @typedef Function
     * @brief The function type supported by `ProgressThread`, returning the progress
     * state of the function.
     */
    using Function = std::function<ProgressState()>;

    /**
     * @brief Store state of a function.
     */
    class FunctionState {
      public:
        /**
         * @brief Construct state of a function.
         *
         * @param function The function to execute.
         */
        explicit FunctionState(Function&& function);

        /**
         * @brief Execute the function.
         *
         * @note Calling this from multiple threads is not allowed.
         */
        void operator()();

        Function function;  ///< The function to execute.
        bool is_done{false};  ///< Whether the function has completed
    };

    /**
     * @brief Construct a new progress thread that can handle multiple functions.
     *
     * @param statistics The statistics instance to use (disabled by default).
     * @param sleep The duration to sleep between each progress loop iteration.
     * If 0, the thread yields execution instead of sleeping. Anecdotally, a 1 us
     * sleep time (the default) is sufficient to avoid starvation and get smooth
     * progress.
     */
    ProgressThread(
        std::shared_ptr<Statistics> statistics = Statistics::disabled(),
        Duration sleep = std::chrono::microseconds{1}
    );

    ~ProgressThread();

    /**
     * @brief Stop the thread, blocking until all functions are done.
     */
    void stop();

    /**
     * @brief Insert a function to process as part of the event loop.
     *
     * @note This function does not need to be thread-safe if not used in
     * multiple progress threads.
     *
     * @param function The function to register.
     *
     * @return The unique ID of the function that was registered.
     */
    FunctionID add_function(Function&& function);

    /**
     * @brief Remove a function and stop processing it as part of the event loop.
     *
     * This function blocks until the function is done (returning `ProgressState::Done`).
     *
     * @param function_id The unique function ID returned by `add_function`.
     *
     * @throws std::logic_error if the function was not registered with this
     * `ProgressThread` or was already removed.
     */
    void remove_function(FunctionID function_id);

    /**
     * @brief Pause the progress thread.
     *
     * @note This blocks until the thread is actually paused.
     */
    void pause();

    /**
     * @brief Resume the progress thread.
     */
    void resume();

    /**
     * @brief Check if the progress thread is currently running.
     *
     * @return true if the thread is running, false otherwise.
     */
    bool is_running() const;

    /**
     * @brief @return The statistics instance on this progress thread.
     */
    std::shared_ptr<Statistics> statistics() const noexcept;

    /**
     * @brief Enable bounded recording of completed collective receives.
     *
     * Enabling starts a new recording interval: queued events and the dropped-event
     * counter are reset.
     *
     * @param capacity Maximum number of queued records.
     */
    void enable_transfer_events(std::size_t capacity = 65536);

    /**
     * @brief Disable recording while preserving queued events for draining.
     */
    void disable_transfer_events();

    /**
     * @brief Move all queued receive records out of the recorder.
     *
     * @return The records queued since the previous drain.
     */
    [[nodiscard]] std::vector<TransferEvent> drain_transfer_events();

    /**
     * @brief @return Number of records dropped in the current recording interval.
     */
    [[nodiscard]] std::uint64_t dropped_transfer_events() const noexcept;

    /**
     * @brief Record a completed collective receive when recording is enabled.
     *
     * Self-transfers and records with no metadata or payload are ignored.
     */
    void record_transfer_event(
        std::int32_t op_id,
        CollectiveKind collective_kind,
        std::int32_t source_rank,
        std::int32_t destination_rank,
        std::uint64_t message_id,
        std::uint64_t metadata_bytes,
        std::uint64_t payload_bytes,
        MemoryType destination_memory_type
    );

  private:
    /**
     * @brief The event loop progressing each of the functions.
     *
     * The event loop continuously progresses registered functions in no
     * specific order.
     */
    void event_loop();

    std::shared_ptr<Statistics> statistics_;
    bool active_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    FunctionIndex next_function_id_{0};
    std::unordered_map<FunctionIndex, FunctionState> functions_;

    std::atomic<bool> transfer_events_enabled_{false};
    std::atomic<std::uint64_t> transfer_events_generation_{0};
    mutable std::mutex transfer_events_mutex_;
    std::vector<TransferEvent> transfer_events_;
    std::size_t transfer_events_capacity_{0};
    std::atomic<std::uint64_t> dropped_transfer_events_{0};

    // Keep `thread_` as the last member so the progress thread cannot run
    // before all other members have been fully initialized.
    detail::PausableThreadLoop thread_;
};

static_assert(StatisticsProvider<ProgressThread>);

}  // namespace rapidsmpf
