/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <utility>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/reservation_attributes.hpp>

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
    MemoryReservation() noexcept = default;

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

    /**
     * @brief Get the telemetry identity carried across moves.
     *
     * @return Reservation identifier, or zero for an unrecorded empty reservation.
     */
    [[nodiscard]] constexpr std::uint64_t id() const noexcept {
        return id_;
    }

    /**
     * @brief Get immutable telemetry attributes inherited by split children.
     *
     * @return Shared reservation attributes.
     */
    [[nodiscard]] ReservationAttributes const& attributes() const noexcept {
        return attrs_;
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
     * @param id Reservation telemetry identity.
     * @param attrs Immutable telemetry attributes.
     */
    MemoryReservation(
        MemoryType mem_type,
        BufferResource* br,
        std::size_t size,
        std::uint64_t id = 0,
        ReservationAttributes attrs = {}
    )
        : mem_type_{mem_type}, br_{br}, size_{size}, id_{id}, attrs_{std::move(attrs)} {}

  private:
    MemoryType mem_type_{MemoryType::HOST};
    BufferResource* br_{nullptr};
    std::size_t size_{0};
    std::uint64_t id_{0};
    ReservationAttributes attrs_;
};
}  // namespace rapidsmpf
