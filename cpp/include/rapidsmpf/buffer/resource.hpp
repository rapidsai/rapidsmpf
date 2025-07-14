/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/spill_manager.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

/**
 * @brief Represents a reservation for future memory allocation.
 *
 * A reservation is returned by `BufferResource::reserve` and must be used when allocating
 * buffers through the `BufferResource`.
 */
class MemoryReservation {
    friend class BufferResource;

  public:
    /**
     * @brief Destructor for the memory reservation.
     *
     * Cleans up resources associated with the reservation.
     */
    ~MemoryReservation() noexcept;

    /**
     * @brief Move constructor for MemoryReservation.
     *
     * @param o The memory reservation to move from.
     */
    MemoryReservation(MemoryReservation&& o)
        : MemoryReservation{
            o.mem_type_, std::exchange(o.br_, nullptr), std::exchange(o.size_, 0)
        } {}

    /**
     * @brief Move assignment operator for MemoryReservation.
     *
     * @param o The memory reservation to move from.
     * @return A reference to the updated MemoryReservation.
     */
    MemoryReservation& operator=(MemoryReservation&& o) noexcept {
        mem_type_ = o.mem_type_;
        br_ = std::exchange(o.br_, nullptr);
        size_ = std::exchange(o.size_, 0);
        return *this;
    }

    /// @brief A memory reservation is not copyable.
    MemoryReservation(MemoryReservation const&) = delete;
    MemoryReservation& operator=(MemoryReservation const&) = delete;

    /**
     * @brief Get the remaining size of the reserved memory.
     *
     * @return The size of the reserved memory in bytes.
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return size_;
    }

    /**
     * @brief Get the type of memory associated with this reservation.
     *
     * @return The type of memory associated with this reservation.
     */
    [[nodiscard]] constexpr MemoryType mem_type() const noexcept {
        return mem_type_;
    }

    /**
     * @brief Get the buffer resource associated with this reservation.
     *
     * @return The buffer resource associated with this reservation.
     */
    [[nodiscard]] constexpr BufferResource* br() const noexcept {
        return br_;
    }

  private:
    /**
     * @brief Constructs a memory reservation.
     *
     * This is private thus only the friend `BufferResource` can create reservations.
     *
     * @param mem_type The type of memory associated with this reservation.
     * @param br Pointer to the buffer resource managing this reservation.
     * @param size The size of the reserved memory in bytes.
     */
    constexpr MemoryReservation(MemoryType mem_type, BufferResource* br, std::size_t size)
        : mem_type_{mem_type}, br_{br}, size_{size} {}

  private:
    MemoryType mem_type_;  ///< The type of memory for this reservation.
    BufferResource* br_;  ///< The buffer resource that manages this reservation.
    std::size_t size_;  ///< The remaining size of the reserved memory in bytes.
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
     * @param memory_available Optional memory availability functions mapping memory types
     * to available memory checkers. Memory types without availability functions are
     * assumed to have unlimited memory.
     * @param periodic_spill_check Enable periodic spill checks. A dedicated thread
     * continuously checks and perform spilling based on the memory availability
     * functions. The value of `periodic_spill_check` is used as the pause between checks.
     * If `std::nullopt`, no periodic spill check is performed.
     * @param statistics The statistics instance to use (disabled by default).
     */
    BufferResource(
        rmm::device_async_resource_ref device_mr,
        std::unordered_map<MemoryType, MemoryAvailable> memory_available = {},
        std::optional<Duration> periodic_spill_check = std::chrono::milliseconds{1},
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
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
     * @brief Get a reference to the current reserved memory of the specified memory type.
     *
     * @param mem_type The target memory type.
     * @return A reference to the memory reserved.
     */
    [[nodiscard]] std::size_t& memory_reserved(MemoryType mem_type) {
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
        MemoryType mem_type, size_t size, bool allow_overbooking
    );

    /**
     * @brief Consume a portion of the reserved memory.
     *
     * Reduces the remaining size of the reserved memory by the specified amount.
     *
     * @param reservation The reservation to release.
     * @param target The memory type of the reservation.
     * @param size The size to consume in bytes.
     * @return The remaining size of the reserved memory after consumption.
     *
     * @throws std::invalid_argument if the memory type does not match the reservation.
     * @throws std::overflow_error if the released size exceeds the size of the
     * reservation.
     */
    std::size_t release(
        MemoryReservation& reservation, MemoryType target, std::size_t size
    );

    /**
     * @brief Allocate a buffer of the specified memory type.
     *
     * @param mem_type The target memory type.
     * @param size The size of the buffer in bytes.
     * @param stream CUDA stream to use for device allocations.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the allocated Buffer.
     *
     * @throws std::invalid_argument if the memory type does not match the reservation.
     * @throws std::overflow_error if `size` exceeds the size of the reservation.
     */
    std::unique_ptr<Buffer> allocate(
        MemoryType mem_type,
        std::size_t size,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

    /**
     * @brief Move host vector data into a Buffer.
     *
     * @param data A unique pointer to the vector containing host data.
     * @return A unique pointer to the resulting Buffer.
     */
    std::unique_ptr<Buffer> move(std::unique_ptr<std::vector<uint8_t>> data);

    /**
     * @brief Move device buffer data into a Buffer.
     *
     * @param data A unique pointer to the device buffer.
     * @param stream CUDA stream used for the data allocation, copy, and/or move.
     * @param event The event to use for the buffer.
     * @return A unique pointer to the resulting Buffer.
     */
    std::unique_ptr<Buffer> move(
        std::unique_ptr<rmm::device_buffer> data,
        rmm::cuda_stream_view stream,
        std::shared_ptr<Buffer::Event> event = nullptr
    );

    /**
     * @brief Move a Buffer to the specified memory type.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param target The target memory type.
     * @param buffer The buffer to move.
     * @param stream CUDA stream used for the buffer allocation, copy, and/or move.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the moved Buffer.
     *
     * @throws std::invalid_argument if `target` does not match the reservation.
     * @throws std::overflow_error if the memory requirement exceeds the reservation.
     */
    std::unique_ptr<Buffer> move(
        MemoryType target,
        std::unique_ptr<Buffer> buffer,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

    /**
     * @brief Move a Buffer to a device buffer.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param buffer The buffer to move.
     * @param stream CUDA stream used for the buffer allocation, copy, and/or move.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the resulting device buffer.
     *
     * @throws std::invalid_argument if the required memory type does not match the
     * reservation.
     * @throws std::overflow_error if the memory requirement exceeds the reservation.
     */
    std::unique_ptr<rmm::device_buffer> move_to_device_buffer(
        std::unique_ptr<Buffer> buffer,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

    /**
     * @brief Move a Buffer to a host vector.
     *
     * If and only if moving between different memory types will this perform a copy.
     *
     * @param buffer The buffer to move.
     * @param stream CUDA stream used for the buffer allocation, copy, and/or move.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the resulting host vector.
     *
     * @throws std::invalid_argument if the required memory type does not match the
     * reservation.
     * @throws std::overflow_error if the memory requirement exceeds the reservation.
     */
    std::unique_ptr<std::vector<uint8_t>> move_to_host_vector(
        std::unique_ptr<Buffer> buffer,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

    /**
     * @brief Create a copy of a Buffer in the specified memory type.
     *
     * Unlike `move()`, this always performs a copy operation.
     *
     * @param target The target memory type.
     * @param buffer The buffer to copy.
     * @param stream CUDA stream used for the buffer allocation and copy.
     * @param reservation The reservation to use for memory allocations.
     * @return A unique pointer to the new Buffer.
     *
     * @throws std::invalid_argument if `target` does not match the reservation.
     * @throws std::overflow_error if the size exceeds the size of the reservation.
     */
    std::unique_ptr<Buffer> copy(
        MemoryType target,
        std::unique_ptr<Buffer> const& buffer,
        rmm::cuda_stream_view stream,
        MemoryReservation& reservation
    );

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

    /**
     * @brief Allocate an empty host buffer.
     *
     * @return A unique pointer to the allocated Buffer.
     */
    std::unique_ptr<Buffer> allocate_empty_host_buffer() const;

  private:
    std::mutex mutex_;
    rmm::device_async_resource_ref device_mr_;
    std::unordered_map<MemoryType, MemoryAvailable> memory_available_;
    // Zero initialized reserved counters.
    std::array<std::size_t, MEMORY_TYPES.size()> memory_reserved_ = {};
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
 * @brief Make a memory reservation or fail.
 *
 * @param br The buffer resource.
 * @param size The size of the buffer to allocate.
 * @param preferred_mem_type The preferred memory type to allocate the buffer from.
 * @return A memory reservation.
 * @throw std::runtime_error if no memory reservation was made.
 */
MemoryReservation reserve_or_fail(
    BufferResource* br,
    size_t size,
    std::optional<MemoryType> const& preferred_mem_type = std::nullopt
);

}  // namespace rapidsmpf
