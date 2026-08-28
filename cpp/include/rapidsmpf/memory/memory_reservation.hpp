/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_pool.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>

namespace rapidsmpf {

class BufferResource;

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
     * @brief Clear the remaining size of the reservation.
     */
    void clear() noexcept;

    /**
     * @brief Split off a sub-reservation of @p size bytes.
     *
     * Reduces this reservation by @p size and returns a new reservation of that size
     * on the same buffer resource and memory type. The total reserved by the buffer
     * resource is unchanged, the bytes are only moved between the two reservations.
     *
     * Useful for scoping part of a reservation to the allocation it covers, since the
     * returned reservation releases its bytes when it goes out of scope.
     *
     * @param size The number of bytes to split off.
     * @return The new reservation.
     *
     * @throws rapidsmpf::reservation_error if @p size exceeds the remaining size.
     */
    [[nodiscard]] MemoryReservation split(std::size_t size);

    /**
     * @brief Move constructor for MemoryReservation.
     *
     * @param o The memory reservation to move from.
     */
    MemoryReservation(MemoryReservation&& o);

    /**
     * @brief Move assignment operator for MemoryReservation.
     *
     * @param o The memory reservation to move from.
     * @return A reference to the updated MemoryReservation.
     */
    MemoryReservation& operator=(MemoryReservation&& o) noexcept;

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
}  // namespace rapidsmpf
