/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <mutex>
#include <optional>

#include <rapidsmpf/pausable_thread_loop.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

namespace detail {
class BufferResourceImpl;
}  // namespace detail

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
     * A spill function takes a requested spill amount as input and returns the actual
     * amount of memory (in bytes) that was spilled.
     *
     * @warning Spill functions must NOT capture an owning reference (shared_ptr, a
     * `BufferResource` by value, etc.) to the `BufferResource` that owns this
     * `SpillManager`. Doing so closes a reference cycle: the BR holds the SpillManager
     * which holds the function which owns the BR. Capture raw pointers/references only,
     * and unregister via `remove_spill_function` from the owner's destructor.
     */
    using SpillFunction = std::function<std::size_t(std::size_t)>;

    /**
     * @brief Represents a unique identifier for a registered spill function.
     */
    using SpillFunctionID = std::size_t;

    /**
     * @brief Constructs a SpillManager instance.
     *
     * @param br_impl Buffer-resource impl used to retrieve current available memory.
     * @param periodic_spill_check Enable periodic spill checks. A dedicated thread
     * continuously checks and perform spilling based on the current available memory as
     * reported by the buffer resource. The value of `periodic_spill_check` is used as the
     * pause between checks. If `std::nullopt`, no periodic spill check is performed.
     */
    SpillManager(
        detail::BufferResourceImpl* br_impl,
        std::optional<Duration> periodic_spill_check = std::nullopt
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
     * available.
     *
     * This method checks the currently available memory and, if insufficient, triggers
     * spilling mechanisms to free up space. Spilling is performed in order of the
     * function priorities until the required headroom is reached or no more spilling is
     * possible.
     *
     * @param headroom The target amount of headroom (in bytes). A negative headroom is
     * allowed and can be used to only trigger spilling when the available memory becomes
     * negative (as reported by the memory resource).
     * @return The actual amount of memory spilled (in bytes), which may be less than
     * requested if there is insufficient spillable data, but may also be more
     * or equal to requested depending on the sizes of spillable data buffers.
     */
    std::size_t spill_to_make_headroom(std::int64_t headroom = 0);

  private:
    mutable std::mutex mutex_;
    std::size_t spill_function_id_counter_{0};
    std::map<SpillFunctionID, SpillFunction> spill_functions_;
    std::multimap<int, SpillFunctionID, std::greater<>> spill_function_priorities_;
    std::optional<detail::PausableThreadLoop> periodic_spill_thread_;

    /// @brief Non-owning back-pointer to the owning impl.
    ///
    /// `BufferResource` is a thin `shared_resource<Impl>` handle: copies share
    /// the same impl but each carries a different `this`, so there is no
    /// canonical outer pointer to point at. The impl IS the resource identity.
    ///
    /// We cannot hold a `BufferResource` by value either that would close a
    /// refcount cycle `BR → Impl → SpillManager → BR`.
    ///
    /// A raw pointer is safe because `SpillManager` is a member of the impl,
    /// so the pointer is stable for `SpillManager`'s entire lifetime.
    detail::BufferResourceImpl* br_impl_;
};


}  // namespace rapidsmpf
