/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

#include <rapidsmpf/pausable_thread_loop.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

class BufferResource;

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
     * @brief Keeps an amount of extra headroom in the periodic spill target while alive.
     */
    class HeadroomToken {
      public:
        /// @brief Constructs an empty token that contributes nothing.
        HeadroomToken() = default;

        ~HeadroomToken();

        HeadroomToken(HeadroomToken const&) = delete;
        HeadroomToken& operator=(HeadroomToken const&) = delete;

        /**
         * @brief Move constructor. Leaves @p o empty.
         *
         * @param o The token to move from.
         */
        HeadroomToken(HeadroomToken&& o) noexcept;

        /**
         * @brief Move assignment. Releases any bytes held by this token first.
         *
         * @param o The token to move from.
         * @return Reference to this token.
         */
        HeadroomToken& operator=(HeadroomToken&& o) noexcept;

        /**
         * @brief The number of bytes this token contributes to the target.
         *
         * @return The byte count, or zero for an empty token.
         */
        [[nodiscard]] std::size_t size() const noexcept {
            return bytes_;
        }

      private:
        friend class SpillManager;

        HeadroomToken(
            std::shared_ptr<std::atomic<std::int64_t>> target, std::size_t bytes
        );

        void release() noexcept;

        std::shared_ptr<std::atomic<std::int64_t>> target_{nullptr};
        std::size_t bytes_{0};
    };

    /**
     * @brief Asks the periodic spill thread to keep @p bytes of headroom available.
     *
     * The periodic thread normally only spills once the memory available for reservation
     * has gone negative, which means spilling happens after the memory limit has already
     * been exceeded. Callers that know they are about to need memory, such as a queued
     * memory reservation request, can use this to have it freed up front instead.
     *
     * The target is the sum of all outstanding tokens, so unrelated subsystems compose.
     * Each caller decides how much to ask for, and the manager does not cap it.
     *
     * The target is **best effort**. It is not a guarantee or a reservation, nothing
     * reads it as an invariant, and a momentarily stale value simply means one check
     * spills a little too much or too little.
     *
     * @note Only device memory is considered, matching `spill_to_make_headroom()`.
     *
     * @note Has no effect when periodic spill checks are disabled, since there is no
     * thread to act on the target.
     *
     * @param bytes The amount of headroom to ask for.
     * @return A token that removes its contribution when destroyed.
     */
    [[nodiscard]] HeadroomToken add_extra_headroom(std::size_t bytes);

    /**
     * @brief The current extra headroom target, the sum of all outstanding tokens.
     *
     * A snapshot, since tokens are created and destroyed concurrently.
     *
     * @return The target in bytes, or zero when no token is outstanding.
     */
    [[nodiscard]] std::int64_t extra_headroom() const noexcept;

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

  private:
    /**
     * @brief Spills memory without locking. The caller must hold `mutex_`.
     *
     * @param amount The amount of memory (in bytes) to spill.
     * @return The actual amount of memory spilled (in bytes).
     */
    std::size_t spill_unsafe(std::size_t amount);

    /**
     * @brief Spills to reach the requested headroom without locking, reading the
     * available memory under the caller's lock. The caller must hold `mutex_`.
     *
     * @param headroom The target amount of headroom (in bytes).
     * @return The actual amount of memory spilled (in bytes).
     */
    std::size_t spill_to_make_headroom_unsafe(std::int64_t headroom);

    mutable std::mutex mutex_;
    BufferResource* br_;
    std::size_t spill_function_id_counter_{0};
    std::map<SpillFunctionID, SpillFunction> spill_functions_;
    std::multimap<int, SpillFunctionID, std::greater<>> spill_function_priorities_;
    std::shared_ptr<std::atomic<std::int64_t>> extra_headroom_{
        std::make_shared<std::atomic<std::int64_t>>(0)
    };
    std::optional<detail::PausableThreadLoop> periodic_spill_thread_;
};


}  // namespace rapidsmpf
