/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>
#include <vector>

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/pausable_thread_loop.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

class BufferResource;
class SpillManager;

/// @brief Opaque process-local attribution token supplied by a telemetry consumer.
using SpillAttributionToken = std::uint64_t;

/// @brief Mutually exclusive reason that a spill attempt was started.
enum class SpillReason : std::uint8_t {
    PERIODIC,
    RESERVATION,
    EAGER,
    EXPLICIT,
};

/**
 * @brief Diagnostic record for one request to spill memory.
 *
 * An attempt can produce zero or more transfer records. `requested_bytes` is a request,
 * not a measurement of bytes copied or memory released.
 */
struct SpillAttemptRecord {
    std::uint64_t event_id{};  ///< Monotonically increasing collector sequence ID.
    std::uint64_t attempt_id{};  ///< ID shared by the attempt and its transfers.
    std::int64_t start_ns{};  ///< Attempt start on the steady-clock nanosecond scale.
    std::size_t requested_bytes{};  ///< Bytes the caller requested to spill.
    SpillReason reason{SpillReason::EXPLICIT};  ///< Exclusive reason for the attempt.
    bool is_background{};  ///< Whether background work initiated the attempt.
    std::optional<SpillAttributionToken> evictor{};  ///< Optional initiating-work token.
};

/**
 * @brief Record for one spill-scoped device-to-host copy submission.
 *
 * A successful record means the asynchronous copy was submitted. It does not mean GPU
 * execution completed or that the source allocation was released. The timestamps use
 * `std::chrono::steady_clock`'s process-local epoch and measure CPU submission time.
 */
struct SpillTransferRecord {
    std::uint64_t event_id{};  ///< Monotonically increasing collector sequence ID.
    std::uint64_t attempt_id{};  ///< ID of the attempt that caused this transfer.
    std::int64_t submission_start_ns{};  ///< CPU submission start, in steady-clock ns.
    std::int64_t submission_end_ns{};  ///< CPU submission end, in steady-clock ns.
    std::size_t submitted_bytes{};  ///< Bytes submitted, or zero on submission failure.
    MemoryType source{MemoryType::DEVICE};  ///< Source memory type.
    MemoryType destination{MemoryType::HOST};  ///< Destination memory type.
    SpillReason reason{SpillReason::EXPLICIT};  ///< Parent attempt's reason.
    bool is_background{};  ///< Whether background work initiated the parent attempt.
    std::optional<SpillAttributionToken> evictor{};  ///< Optional initiating-work token.
    std::optional<SpillAttributionToken> buffer_owner{};  ///< Optional data-owner token.
    bool is_success{};  ///< Whether the asynchronous copy was successfully submitted.
};

/// @brief A collected spill attempt or transfer event.
using SpillEvent = std::variant<SpillAttemptRecord, SpillTransferRecord>;

/**
 * @brief Bounded thread-safe collector for spill telemetry.
 *
 * Event IDs are monotonically increasing and reads use the half-open range
 * `[begin_sequence, end_sequence)`. When the collector is full, the oldest event is
 * overwritten and `dropped_events()` is incremented. Consumers must treat a nonzero
 * dropped count as incomplete telemetry.
 *
 * Collection is disabled by default. Enabling preallocates all producer-side storage;
 * recording while disabled is allocation-free.
 */
class SpillEventCollector {
  public:
    /**
     * @brief Enable collection with bounded preallocated storage.
     *
     * Existing collected events and the dropped-event count are cleared. Sequence IDs
     * remain monotonic across calls.
     *
     * @param capacity Maximum number of retained events; must be greater than zero.
     */
    void enable(std::size_t capacity);

    /**
     * @brief Disable collection and release retained event storage.
     */
    void disable() noexcept;

    /**
     * @brief Check whether event collection is enabled.
     *
     * @return True when event collection is enabled.
     */
    [[nodiscard]] bool enabled() const noexcept;

    /**
     * @brief Get the exclusive sequence cursor for events recorded so far.
     *
     * @return The sequence ID that will be assigned to the next retained event.
     */
    [[nodiscard]] std::uint64_t sequence() const noexcept;

    /**
     * @brief Get the number of events lost because bounded storage was full.
     *
     * @return Number of dropped or overwritten events since the last `enable()`.
     */
    [[nodiscard]] std::uint64_t dropped_events() const noexcept;

    /**
     * @brief Read all event types in a half-open sequence range.
     *
     * @param begin_sequence First sequence ID to include.
     * @param end_sequence Exclusive sequence ID at which to stop.
     * @return Retained events in sequence order.
     */
    [[nodiscard]] std::vector<SpillEvent> read(
        std::uint64_t begin_sequence, std::uint64_t end_sequence
    ) const;

    /**
     * @brief Read spill attempts in a half-open sequence range.
     *
     * @param begin_sequence First sequence ID to include.
     * @param end_sequence Exclusive sequence ID at which to stop.
     * @return Retained attempt records in sequence order.
     */
    [[nodiscard]] std::vector<SpillAttemptRecord> read_attempts(
        std::uint64_t begin_sequence, std::uint64_t end_sequence
    ) const;

    /**
     * @brief Read spill transfers in a half-open sequence range.
     *
     * @param begin_sequence First sequence ID to include.
     * @param end_sequence Exclusive sequence ID at which to stop.
     * @return Retained transfer records in sequence order.
     */
    [[nodiscard]] std::vector<SpillTransferRecord> read_transfers(
        std::uint64_t begin_sequence, std::uint64_t end_sequence
    ) const;

  private:
    friend class SpillManager;
    void record(SpillEvent event) noexcept;

    mutable std::mutex mutex_;
    std::vector<SpillEvent> events_;
    std::size_t oldest_index_{};
    std::size_t size_{};
    std::atomic<bool> enabled_{false};
    std::atomic<std::uint64_t> next_sequence_{1};
    std::atomic<std::uint64_t> dropped_events_{0};
};

/**
 * @brief RAII override for the owner of transfers in the current spill callback.
 *
 * Nested scopes take precedence and destruction restores the previous owner. This is
 * primarily used by SpillManager registrations; mixed-owner stores may use a narrower
 * scope around an individual item copy.
 */
class SpillBufferOwnerScope {
  public:
    /**
     * @brief Override the buffer owner for the current spill attempt.
     *
     * @param manager Spill manager that owns the active attempt.
     * @param buffer_owner Optional owner token to install for this scope.
     */
    SpillBufferOwnerScope(
        SpillManager& manager, std::optional<SpillAttributionToken> buffer_owner
    );
    ~SpillBufferOwnerScope();
    SpillBufferOwnerScope(SpillBufferOwnerScope const&) = delete;
    SpillBufferOwnerScope& operator=(SpillBufferOwnerScope const&) = delete;

  private:
    SpillManager& manager_;
    std::optional<SpillAttributionToken> previous_;
    bool has_active_attempt_{false};
};

/**
 * @brief Manages memory spilling to free up device memory when needed.
 *
 * The SpillManager is responsible for registering, prioritizing, and executing spill
 * functions to ensure efficient memory management.
 */
class SpillManager {
  public:
    /**
     * @brief Spill function type.
     *
     * A spill function receives a requested spill size in bytes and returns the actual
     * number of bytes spilled.
     *
     * Spill functions must not capture owning references to the `BufferResource` that
     * owns this `SpillManager`, either directly or indirectly through objects that
     * allocate from `BufferResource::device_mr()`. Doing so creates a reference cycle:
     *
     * `BufferResource -> SpillManager -> SpillFunction -> BufferResource`
     *
     * Spill functions should capture only non-owning references or raw pointers. Owners
     * registering spill functions should additionally call `remove_spill_function()`
     * during destruction before releasing any buffer-owning members.
     */
    using SpillFunction = std::function<std::size_t(std::size_t)>;

    /**
     * @brief Represents a unique identifier for a registered spill function.
     */
    using SpillFunctionID = std::size_t;

    /**
     * @brief Constructs a SpillManager instance.
     *
     * @param br Buffer resource used to retrieve current available memory.
     * @param periodic_spill_check Enable periodic spill checks. A dedicated thread
     * continuously checks and perform spilling based on the current available memory as
     * reported by the buffer resource. The value of `periodic_spill_check` is used as the
     * pause between checks. If `std::nullopt`, no periodic spill check is performed.
     */
    SpillManager(
        BufferResource* br, std::optional<Duration> periodic_spill_check = std::nullopt
    );

    /**
     * @brief Destructor for SpillManager.
     *
     * Cleans up any allocated resources and stops periodic spill checks if active (this
     * will block until all spill functions has stopped).
     */
    ~SpillManager();

    /**
     * @brief Adds a spill function with a given priority to the spill manager.
     *
     * The spill function is prioritized according to the specified priority value.
     *
     * @param spill_function The spill function to be added.
     * @param priority The priority level of the spill function (higher values indicate
     * higher priority).
     * @return The id assigned to the newly added spill function.
     */
    SpillFunctionID add_spill_function(SpillFunction spill_function, int priority);

    /**
     * @brief Adds a spill function with priority and optional buffer ownership.
     *
     * @param spill_function The spill function to be added.
     * @param priority Higher values run first.
     * @param buffer_owner Optional opaque owner token inherited by callback transfers.
     * @return The id assigned to the newly added spill function.
     */
    SpillFunctionID add_spill_function(
        SpillFunction spill_function,
        int priority,
        std::optional<SpillAttributionToken> buffer_owner
    );

    /**
     * @brief Removes a spill function from the spill manager.
     *
     * This method unregisters the spill function associated with the given ID and removes
     * it from the priority list. If no more spill functions remain, the periodic spill
     * thread is paused.
     *
     * @param fid The id of the spill function to be removed.
     */
    void remove_spill_function(SpillFunctionID fid);

    /**
     * @brief Initiates spilling to free up a specified amount of memory.
     *
     * This method iterates through registered spill functions in priority order, invoking
     * them until at least the requested amount of memory has been spilled or no more
     * spilling is possible.
     *
     * @param amount The amount of memory (in bytes) to spill.
     * @return The actual amount of memory spilled (in bytes), which may be more, less
     * or equal to the requested.
     */
    std::size_t spill(std::size_t amount);

    /**
     * @brief Initiates an attributed foreground spill.
     *
     * @param amount Number of bytes requested.
     * @param reason Exclusive reason for the spill.
     * @param evictor Optional opaque token for the work requesting the spill.
     * @return The amount reported by spill callbacks.
     */
    std::size_t spill(
        std::size_t amount,
        SpillReason reason,
        std::optional<SpillAttributionToken> evictor
    );

    /**
     * @brief Attempts to free up memory by spilling data until the requested headroom is
     * reservable.
     *
     * The headroom measurement is a snapshot, so a later `reserve()` of `headroom` bytes
     * is not guaranteed to succeed. Spilling is performed in order of the function
     * priorities until the requested headroom is reservable or no more spilling is
     * possible. Spilling reduces allocations, never outstanding reservations.
     *
     * @param headroom The target amount of headroom (in bytes). A negative headroom
     * triggers spilling only once the memory available for reservation drops below
     * `headroom`.
     * @return The actual amount of memory spilled (in bytes), which may be less than
     * requested if there is insufficient spillable data, but may also be more
     * or equal to requested depending on the sizes of spillable data buffers.
     *
     * @see BufferResource::memory_available_for_reservation()
     */
    std::size_t spill_to_make_headroom(std::int64_t headroom = 0);

    /**
     * @brief Attributed form of `spill_to_make_headroom()`.
     *
     * @param headroom Target reservable device-memory headroom.
     * @param reason Exclusive reason for the spill.
     * @param evictor Optional opaque token for the work requesting the spill.
     * @param is_background Whether background work initiated this request.
     * @return The amount reported by spill callbacks.
     */
    std::size_t spill_to_make_headroom(
        std::int64_t headroom,
        SpillReason reason,
        std::optional<SpillAttributionToken> evictor,
        bool is_background
    );

    /**
     * @brief Non-blocking version of `spill_to_make_headroom()`.
     *
     * Returns immediately instead of waiting when the spill lock is unavailable.
     * Intended for pollers that retry, such as the streaming layer's memory
     * reservation loop.
     *
     * @param headroom The target amount of headroom (in bytes). A negative headroom
     * triggers spilling only once the memory available for reservation drops below
     * `headroom`.
     * @return The actual amount of memory spilled (in bytes), or `std::nullopt` if no
     * spill was attempted. A `std::nullopt` result does not imply that spilling is
     * impossible or that another spill is in progress. Callers should retry.
     *
     * @see spill_to_make_headroom()
     */
    std::optional<std::size_t> try_spill_to_make_headroom(std::int64_t headroom = 0);

    /**
     * @brief Access the collector owned by this manager's BufferResource.
     *
     * @return Mutable reference to the spill event collector.
     */
    [[nodiscard]] SpillEventCollector& event_collector() noexcept;

    /**
     * @brief Access the collector owned by this manager's BufferResource.
     *
     * @return Const reference to the spill event collector.
     */
    [[nodiscard]] SpillEventCollector const& event_collector() const noexcept;

    /**
     * @brief Record a copy submission against this manager's active spill attempt.
     *
     * This function is non-throwing and ignores copies outside a spill attempt as well as
     * copies other than DEVICE to PINNED_HOST/HOST.
     *
     * @param source Source memory type.
     * @param destination Destination memory type.
     * @param bytes Number of bytes requested for the copy.
     * @param submission_start_ns CPU submission start on the steady-clock nanosecond
     * scale.
     * @param submission_end_ns CPU submission end on the steady-clock nanosecond scale.
     * @param is_success Whether the asynchronous copy was successfully submitted.
     */
    void record_transfer_submission(
        MemoryType source,
        MemoryType destination,
        std::size_t bytes,
        std::int64_t submission_start_ns,
        std::int64_t submission_end_ns,
        bool is_success
    ) noexcept;

  private:
    friend class SpillBufferOwnerScope;

    struct ActiveSpillAttempt {
        std::uint64_t attempt_id;
        SpillReason reason;
        bool is_background;
        std::optional<SpillAttributionToken> evictor;
        std::optional<SpillAttributionToken> buffer_owner;
        std::thread::id thread_id;
    };

    class SpillAttemptScope;

    /**
     * @brief Spills memory without locking. The caller must hold `mutex_`.
     *
     * @param amount The amount of memory (in bytes) to spill.
     * @param reason Exclusive reason for the spill attempt.
     * @param evictor Optional opaque token for the work requesting the spill.
     * @param is_background Whether background work initiated the spill.
     * @return The actual amount of memory spilled (in bytes).
     */
    std::size_t spill_unsafe(
        std::size_t amount,
        SpillReason reason,
        std::optional<SpillAttributionToken> evictor,
        bool is_background
    );

    /**
     * @brief Spills to reach the requested headroom without locking, reading the
     * available memory under the caller's lock. The caller must hold `mutex_`.
     *
     * @param headroom The target amount of headroom (in bytes).
     * @param reason Exclusive reason for the spill attempt.
     * @param evictor Optional opaque token for the work requesting the spill.
     * @param is_background Whether background work initiated the spill.
     * @return The actual amount of memory spilled (in bytes).
     */
    std::size_t spill_to_make_headroom_unsafe(
        std::int64_t headroom,
        SpillReason reason,
        std::optional<SpillAttributionToken> evictor,
        bool is_background
    );

    struct RegisteredSpillFunction {
        SpillFunction function;
        std::optional<SpillAttributionToken> buffer_owner;
    };

    mutable std::mutex mutex_;
    BufferResource* br_;
    std::size_t spill_function_id_counter_{0};
    std::atomic<std::uint64_t> attempt_id_counter_{1};
    std::map<SpillFunctionID, RegisteredSpillFunction> spill_functions_;
    std::multimap<int, SpillFunctionID, std::greater<>> spill_function_priorities_;
    mutable std::mutex active_attempt_mutex_;
    std::optional<ActiveSpillAttempt> active_attempt_;
    SpillEventCollector event_collector_;
    std::optional<detail::PausableThreadLoop> periodic_spill_thread_;
};


}  // namespace rapidsmpf
