/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <limits>
#include <utility>

#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {


MemoryReservation::~MemoryReservation() noexcept {
    if (size_ > 0) {
        br_->release(*this, mem_type_, size_);
    }
}

SpillManager::SpillManager(BufferResource* br) : br_{br} {}

SpillManager::~SpillManager() {}

SpillManager::SpillFunctionID SpillManager::add_spill_function(
    SpillFunction spill_function, int priority
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const id = spill_function_id_counter_++;
    RAPIDSMP_EXPECTS(
        spill_functions_.insert({id, std::move(spill_function)}).second,
        "corrupted id counter"
    );
    spill_function_priorities_.insert({priority, id});
    return id;
}

void SpillManager::remove_spill_function(SpillFunctionID fid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& prio = spill_function_priorities_;
    for (auto it = prio.begin(); it != prio.end(); ++it) {
        if (it->second == fid) {
            prio.erase(it);  // Erase the first occurrence
            break;  // Exit after erasing to ensure only the first one is removed
        }
    }
    spill_functions_.erase(fid);
}

std::size_t SpillManager::spill(std::size_t amount) {
    std::size_t spilled{0};
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto const [_, fid] : spill_function_priorities_) {
        if (spilled >= amount) {
            break;
        }
        spilled += spill_functions_.at(fid)(amount - spilled);
    }
    return spilled;
}

/**
 * @brief Attempts to free up memory by spilling data until the requested headroom is
 * available.
 *
 * This method checks the currently available memory and, if insufficient, triggers
 * spilling mechanisms to free up space. Spilling is performed in order of the function
 * priorities until the required headroom is reached or no more spilling is possible.
 *
 * @param headroom The target amount of headroom (in bytes). Allowed to be negative.
 * @return The actual amount of memory spilled (in bytes), which may be less than
 * requested if there is insufficient spillable data.
 */
std::size_t SpillManager::spill_to_make_headroom(std::int64_t headroom) {
    // TODO: check other memory types.
    std::int64_t available = br_->memory_available(MemoryType::DEVICE)();
    if (headroom <= available) {
        return 0;
    }
    return spill(headroom - available);
}

BufferResource::BufferResource(
    rmm::device_async_resource_ref device_mr,
    std::unordered_map<MemoryType, MemoryAvailable> memory_available
)
    : device_mr_{device_mr},
      memory_available_{std::move(memory_available)},
      spill_manager_{this} {
    for (MemoryType mem_type : MEMORY_TYPES) {
        // Add missing memory availability functions.
        memory_available_.try_emplace(mem_type, std::numeric_limits<std::int64_t>::max);
    }
}

std::pair<MemoryReservation, std::size_t> BufferResource::reserve(
    MemoryType mem_type, std::size_t size, bool allow_overbooking
) {
    auto const& available = memory_available(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t& reserved = memory_reserved(mem_type);

    // Calculate the available memory _after_ the memory has been reserved.
    std::int64_t headroom = available() - (reserved + size);
    // If negative, we are overbooking.
    std::size_t overbooking = headroom < 0 ? -headroom : 0;
    if (overbooking > 0 && !allow_overbooking) {
        // Cancel the reservation, overbooking isn't allowed.
        return {MemoryReservation(mem_type, this, 0), overbooking};
    }
    // Make the reservation.
    reserved += size;
    return {MemoryReservation(mem_type, this, size), overbooking};
}

std::size_t BufferResource::release(
    MemoryReservation& reservation, MemoryType target, std::size_t size
) {
    RAPIDSMP_EXPECTS(
        reservation.mem_type_ == target,
        "the memory type of MemoryReservation doesn't match",
        std::invalid_argument
    );
    std::lock_guard const lock(mutex_);
    RAPIDSMP_EXPECTS(
        size <= reservation.size_,
        "MemoryReservation(" + format_nbytes(reservation.size_) + ") isn't big enough ("
            + format_nbytes(size) + ")",
        std::overflow_error
    );
    std::size_t& reserved = memory_reserved(target);
    RAPIDSMP_EXPECTS(reserved >= size, "corrupted reservation stat");
    reserved -= size;
    return reservation.size_ -= size;
}

std::unique_ptr<Buffer> BufferResource::allocate(
    MemoryType mem_type,
    std::size_t size,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    std::unique_ptr<Buffer> ret;
    switch (mem_type) {
    case MemoryType::HOST:
        // TODO: use pinned memory, maybe use rmm::mr::pinned_memory_resource and
        // std::pmr::vector?
        ret = std::make_unique<Buffer>(
            Buffer{std::make_unique<std::vector<uint8_t>>(size), this}
        );
        break;
    case MemoryType::DEVICE:
        ret = std::make_unique<Buffer>(
            Buffer{std::make_unique<rmm::device_buffer>(size, stream, device_mr_), this}
        );
        break;
    default:
        RAPIDSMP_FAIL("MemoryType: unknown");
    }
    release(reservation, mem_type, size);
    return ret;
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<std::vector<uint8_t>> data, rmm::cuda_stream_view stream
) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    std::unique_ptr<rmm::device_buffer> data, rmm::cuda_stream_view stream
) {
    return std::make_unique<Buffer>(Buffer{std::move(data), this});
}

std::unique_ptr<Buffer> BufferResource::move(
    MemoryType target,
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    if (target != buffer->mem_type) {
        auto ret = buffer->copy(target, stream);
        release(reservation, target, ret->size);
        return ret;
    }
    return buffer;
}

std::unique_ptr<rmm::device_buffer> BufferResource::move_to_device_buffer(
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    return std::move(
        move(MemoryType::DEVICE, std::move(buffer), stream, reservation)->device()
    );
}

std::unique_ptr<std::vector<uint8_t>> BufferResource::move_to_host_vector(
    std::unique_ptr<Buffer> buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    return std::move(
        move(MemoryType::HOST, std::move(buffer), stream, reservation)->host()
    );
}

std::unique_ptr<Buffer> BufferResource::copy(
    MemoryType target,
    std::unique_ptr<Buffer> const& buffer,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    auto ret = buffer->copy(target, stream);
    release(reservation, target, ret->size);
    return ret;
}

}  // namespace rapidsmp
