/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>

#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/host_memory_resource.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief Policy controlling whether a memory reservation is allowed to overbook.
 *
 * This enum is used throughout RapidsMPF to specify the overbooking behavior of
 * a memory reservation request. The exact semantics depend on the specific API
 * and execution context in which it is used.
 */
enum class AllowOverbooking : bool {
    NO,  ///< Overbooking is not allowed.
    YES,  ///< Overbooking is allowed.
};

/**
 * @brief Class managing buffer resources.
 *
 * This class handles memory allocation and transfers between different memory types
 * (e.g., host and device). All memory operations in rapidsmpf, such as those performed
 * by the Shuffler, rely on a buffer resource for memory management.
 *
 * @note Similar to RMM's memory resource, the `BufferResource` instance must outlive all
 * allocated buffers and memory reservations.
 */
class BufferResource {
  public:
    /**
     * @brief Callback function to determine available memory.
     *
     * The function should return the current available memory of a specific type and
     * must be thread-safe iff used by multiple `BufferResource` instances concurrently.
     *
     * @warning Calling any `BufferResource` instance methods in the function might result
     * in a deadlock. This is because the buffer resource is locked when the function is
     * called.
     */
    using MemoryAvailable = std::function<std::int64_t()>;

    /**
     * @brief Constructs a buffer resource.
     *
     * @param device_mr Reference to the RMM device memory resource used for device
     * allocations.
     * @param pinned_mr The pinned host memory resource used for `MemoryType::PINNED_HOST`
     * allocations. If null, pinned host allocations are disabled. In that case, any
     * attempt to allocate pinned memory will fail regardless of what @p memory_available
     * reports.
     * @param memory_available Optional functions that report available memory for each
     * memory type. If a memory type is not present in this map, it is treated as having
     * unlimited available memory. The only exception is `MemoryType::PINNED_HOST`, which
     * is always assigned a zero-capacity function when `pinned_mr` is disabled.
     * @param periodic_spill_check Enable periodic spill checks. A dedicated thread
     * continuously checks and perform spilling based on the memory availability
     * functions. The value of `periodic_spill_check` is used as the pause between checks.
     * If `std::nullopt`, no periodic spill check is performed.
     * @param stream_pool Pool of CUDA streams. Used throughout RapidsMPF for operations
     * that do not take an explicit CUDA stream.
     * @param statistics The statistics instance to use (disabled by default).
     */
    BufferResource(
        rmm::device_async_resource_ref device_mr,
        std::shared_ptr<PinnedMemoryResource> pinned_mr = PinnedMemoryResource::Disabled,
        std::unordered_map<MemoryType, MemoryAvailable> memory_available = {},
        std::optional<Duration> periodic_spill_check = std::chrono::milliseconds{1},
        std::shared_ptr<rmm::cuda_stream_pool> stream_pool = std::make_shared<
            rmm::cuda_stream_pool>(16, rmm::cuda_stream::flags::non_blocking),
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    );

    /**
     * @brief Construct a BufferResource from configuration options.
     *
     * This factory method creates a BufferResource using configuration options to
     * initialize all components.
     *
     * @param mr Pointer to the RMM resource adaptor, which must outlive the
     * returned BufferResource.
     * @param options Configuration options.
     *
     * @return A shared pointer to a BufferResource instance configured according to the
     * options.
     */
    static std::shared_ptr<BufferResource> from_options(
        RmmResourceAdaptor* mr, config::Options options
    );

    ~BufferResource() noexcept = default;

    /**
     * @brief Get the RMM device memory resource.
     *
     * @return Reference to the RMM resource used for device allocations.
     */
    [[nodiscard]] rmm::device_async_resource_ref device_mr() const noexcept {
        return device_mr_;
    }

    /**
     * @brief Get the RMM host memory resource.
     *
     * @return Reference to the RMM resource used for host allocations.
     */
    [[nodiscard]] rmm::host_async_resource_ref host_mr() noexcept {
        return host_mr_;
    }

    /**
     * @brief Get the RMM pinned host memory resource.
     *
     * @return Reference to the RMM resource used for pinned host allocations.
     */
    [[nodiscard]] rmm::host_async_resource_ref pinned_mr() {
        RAPIDSMPF_EXPECTS(
            pinned_mr_, "no pinned memory resource is available", std::invalid_argument
        );
        return *pinned_mr_;
    }

    /**
     * @brief Retrieves the memory availability function for a given memory type.
     *
     * This function returns the callback function used to determine the available memory
     * for the specified memory type.
     *
     * @param mem_type The type of memory whose availability function is requested.
     * @return Reference to the memory availability function associated with `mem_type`.
     */
    [[nodiscard]] MemoryAvailable const& memory_available(MemoryType mem_type) const {
        return memory_available_.at(mem_type);
    }

    /**
     * @brief Get the current reserved memory of the specified memory type.
     *
     * @param mem_type The target memory type.
     * @return The memory reserved.
     */
    [[nodiscard]] std::size_t memory_reserved(MemoryType mem_type) const {
        return memory_reserved_[static_cast<std::size_t>(mem_type)];
    }

    /**
     * @brief Reserve an amount of the specified memory type.
     *
     * Creates a new reservation of the specified size and type to inform about upcoming
     * buffer allocations.
     *
     * If overbooking is allowed, a reservation of `size` is returned even when the amount
     * of memory isn't available. In this case, the caller must promise to free buffers
     * corresponding to (at least) the amount of overbooking before using the reservation.
     *
     * If overbooking isn't allowed, a reservation of size zero is returned on failure.
     *
     * @param mem_type The target memory type.
     * @param size The number of bytes to reserve.
     * @param allow_overbooking Whether overbooking is allowed.
     * @return A pair containing the reservation and the amount of overbooking. On success
     * the size of the reservation always equals `size` and on failure the size always
     * equals zero (a zero-sized reservation never fails).
     */
    std::pair<MemoryReservation, std::size_t> reserve(
        MemoryType mem_type, size_t size, AllowOverbooking allow_overbooking
    );

    /**
     * @brief Reserve device memory and spill if necessary.
     *
     * Attempts to reserve the requested amount of device memory. If insufficient memory
     * is available, spilling is triggered to free up space. When overbooking is allowed,
     * the reservation may succeed even if spilling was not sufficient to fully satisfy
     * the request.
     *
     * @param size The size of the memory to reserve.
     * @param allow_overbooking Whether to allow overbooking. If false, ensures enough
     * memory is freed to satisfy the reservation; otherwise, allows overbooking even
     * if spilling was insufficient.
     * @return The memory reservation.
     *
     * @throws rapidsmpf::reservation_error if allow_overbooking is false and the buffer
     * resource cannot reserve and spill enough device memory.
     */
    MemoryReservation reserve_device_memory_and_spill(
        size_t size, AllowOverbooking allow_overbooking
    );

    /**
     * @brief Make a memory reservation or fail based on the given order of memory types.
     *
     * The function attempts to reserve memory by iterating over @p mem_types in the given
     * order of preference. For each memory type, it requests a reservation without
     * overbooking. If no memory type can satisfy the request, the function throws.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_types Range of memory types to try to reserve memory from.
     * @return A memory reservation.
     *
     * @throws std::runtime_error if no memory reservation was made.
     */
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, MemoryType>
    [[nodiscard]] MemoryReservation reserve_or_fail(size_t size, Range mem_types) {
        // try to reserve memory from the given order
        for (auto const& mem_type : mem_types) {
            if (mem_type == MemoryType::PINNED_HOST
                && pinned_mr_ == PinnedMemoryResource::Disabled)
            {
                // Pinned host memory is only available if the memory resource is
                // available.
                continue;
            }
            auto [res, _] = reserve(mem_type, size, AllowOverbooking::NO);
            if (res.size() == size) {
                return std::move(res);
            }
        }
        RAPIDSMPF_FAIL("failed to reserve memory", std::runtime_error);
    }

    /**
     * @brief Make a memory reservation or fail.
     *
     * @param size The size of the buffer to allocate.
     * @param mem_type The memory type to try to reserve memory from.
     * @return A memory reservation.
     *
     * @throws std::runtime_error if no memory reservation was made.
     */
    [[nodiscard]] MemoryReservation reserve_or_fail(size_t size, MemoryType mem_type) {
        return reserve_or_fail(size, std::ranges::single_view{mem_type});
    }

    /**
     * @brief Consume a portion of the reserved memory.
     *
     * Reduces the remaining size of the reserved memory by the specified amount.
     *
     * @param reservation The reservation to release.
     * @param size The size to consume in bytes.
     * @return The remaining size of the reserved memory after consumption.
     *
     * @throws rapidsmpf::reservation_error if the released size exceeds the size of the
     * reservation.
     */
    std::size_t release(MemoryReservation& reservation, std::size_t size);

    /**
     * @brief Allocate a buffer of the specified memory type by the reservation.
     *
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the allocated Buffer.
     *
     * @throws std::invalid_argument if the memory type does not match the reservation.
     * @throws rapidsmpf::reservation_error if `size` exceeds the size of the reservation.
     */
    std::unique_ptr<Buffer> allocate(
        std::size_t size, rmm::cuda_stream_view stream, MemoryReservation& reservation
    );

    /**
     * @brief Allocate a buffer consuming the entire reservation.
     *
     * This overload allocates a buffer that matches the full size and memory type
     * of the provided reservation. The reservation is consumed by the call.
     *
     * @param stream CUDA stream to use for device allocations.
     * @param reservation The memory reservation to consume for the allocation.
     * @return A unique pointer to the allocated Buffer.
     */
    std::unique_ptr<Buffer> allocate(
        rmm::cuda_stream_view stream, MemoryReservation&& reservation
    );

    /**
     * @brief Move device buffer data into a Buffer.
     *
     * This operation is cheap; no copy is performed. The resulting Buffer resides in
     * device memory.
     *
     * If @p stream differs from the device buffer's current stream:
     *   - @p stream is synchronized with the device buffer's current stream, and
     *   - the device buffer's current stream is updated to @p stream.
     *
     * @param data Unique pointer to the device buffer.
     * @param stream CUDA stream associated with the new Buffer. Use or synchronize with
     * this stream when operating on the Buffer.
     * @return Unique pointer to the resulting Buffer.
     */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
    );

    /**
     * @brief Move a Buffer to the memory type specified by the reservation.
     *
     * If the Buffer already resides in the target memory type, a cheap move is performed.
     * Otherwise, the Buffer is copied to the target memory using its own CUDA stream.
     *
     * @param buffer Buffer to move.
     * @param reservation Memory reservation used if a copy is required.
     * @return Unique pointer to the resulting Buffer.
     *
     * @throws rapidsmpf::reservation_error If the allocation size exceeds the
     * reservation.
     */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    );

    /**
     * @brief Move a Buffer to a device buffer.
     *
     * If the Buffer already resides in device memory, a cheap move is performed.
     * Otherwise, the Buffer is copied to device memory using its own CUDA stream.
     *
     * @param buffer The buffer to move.
     * @param reservation Memory reservation used if a copy is required.
     * @return A unique pointer to the resulting device buffer.
     *
     * @throws std::invalid_argument If the reservation's memory type isn't device memory.
     * @throws rapidsmpf::reservation_error if the memory requirement exceeds the
     * reservation.
     */
    std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    );

    /**
     * @brief Move a Buffer into a host buffer.
     *
     * If the Buffer already resides in host memory, a cheap move is performed.
     * Otherwise, the Buffer is copied to host memory using its own CUDA stream.
     *
     * @param buffer Buffer to move.
     * @param reservation Memory reservation used if a copy is required.
     * @return Unique pointer to the resulting host buffer.
     *
     * @throws std::invalid_argument If the reservation's memory type isn't host memory.
     * @throws rapidsmpf::reservation_error If the allocation size exceeds the
     * reservation.
     */
    std::unique_ptr<HostBuffer> move_to_host_buffer(
        std::unique_ptr<Buffer> buffer, MemoryReservation& reservation
    );

    /**
     * @brief Returns the CUDA stream pool used by this buffer resource.
     *
     * Use this pool for operations that do not take an explicit CUDA stream.
     *
     * @return Reference to the underlying CUDA stream pool.
     */
    rmm::cuda_stream_pool const& stream_pool() const;

    /**
     * @brief Gets a reference to the spill manager used.
     *
     * @return Reference to the SpillManager instance.
     */
    SpillManager& spill_manager();

    /**
     * @brief Gets a shared pointer to the statistics associated with this buffer
     * resource.
     *
     * @return Shared pointer the Statistics instance.
     */
    std::shared_ptr<Statistics> statistics();

  private:
    std::mutex mutex_;
    rmm::device_async_resource_ref device_mr_;
    std::shared_ptr<PinnedMemoryResource> pinned_mr_;
    HostMemoryResource host_mr_;
    std::unordered_map<MemoryType, MemoryAvailable> memory_available_;
    // Zero initialized reserved counters.
    std::array<std::size_t, MEMORY_TYPES.size()> memory_reserved_ = {};
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool_;
    SpillManager spill_manager_;
    std::shared_ptr<Statistics> statistics_;
};

/**
 * @brief A functor for querying the remaining available memory within a defined limit
 * from an RMM statistics resource.
 *
 * This class is designed to be used as a callback to provide available memory
 * information in the context of memory management, such as when working with
 * `BufferResource`. The available memory is determined as the difference
 * between a user-defined limit and the memory currently used, as reported
 * by an RMM statistics resource adaptor.
 *
 * By enforcing a limit, this functor can be used to simulate constrained memory
 * environments or to prevent memory allocation beyond a specific threshold.
 *
 * @see rapidsmpf::BufferResource::MemoryAvailable
 */
class LimitAvailableMemory {
  public:
    /**
     * @brief Constructs a `LimitAvailableMemory` instance.
     *
     * @param mr A pointer to an RMM resource adaptor. The underlying resource
     * adaptor must outlive this instance.
     * @param limit The maximum memory available (in bytes). Used to calculate the
     * remaining memory.
     */
    constexpr LimitAvailableMemory(RmmResourceAdaptor const* mr, std::int64_t limit)
        : limit{limit}, mr_{mr} {}

    /**
     * @brief Returns the remaining available memory within the defined limit.
     *
     * This operator queries the `RmmResourceAdaptor` to determine the memory
     * currently used and calculates the remaining memory as:
     * `limit - used_memory`.
     *
     * @return The remaining memory in bytes.
     */
    std::int64_t operator()() const {
        return limit - static_cast<std::int64_t>(mr_->current_allocated());
    }

  public:
    std::int64_t const limit;  ///< The memory limit.

  private:
    RmmResourceAdaptor const* mr_;
};

/**
 * @brief Construct a map of memory-available functions from configuration options.
 *
 * @param mr Pointer to a memory resource adaptor.
 * @param options Configuration options.
 *
 * @return The map of memory-available functions.
 */
std::unordered_map<MemoryType, BufferResource::MemoryAvailable>
memory_available_from_options(RmmResourceAdaptor* mr, config::Options options);

/**
 * @brief Get the `periodic_spill_check` parameter from configuration options.
 *
 * @param options Configuration options.
 *
 * @return The duration of the pause between spill checks or std::nullopt if no dedicated
 * thread should check for spilling.
 */
std::optional<Duration> periodic_spill_check_from_options(config::Options options);

/**
 * @brief Get a new CUDA stream pool from configuration options.
 *
 * @param options Configuration options.
 * @return Pool of CUDA streams used throughout RapidsMPF for operations that do
 * not take an explicit CUDA stream.
 */
std::shared_ptr<rmm::cuda_stream_pool> stream_pool_from_options(config::Options options);


}  // namespace rapidsmpf
