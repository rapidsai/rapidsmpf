/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

namespace {

/// @brief Throw if data in @p mem_type cannot be spilled, i.e. it isn't addressable.
void expect_spillable(MemoryType mem_type) {
    RAPIDSMPF_EXPECTS(
        contains(ADDRESSABLE_MEMORY_TYPES, mem_type),
        "cannot spill from " + std::string{to_string(mem_type)},
        std::invalid_argument
    );
}

}  // namespace

SpillManager::SpillManager(
    BufferResource* br, std::optional<Duration> periodic_spill_check
)
    : br_{br} {
    if (periodic_spill_check.has_value()) {
        periodic_spill_thread_.emplace(
            // Only the spill thread touches `failures`.
            [this, failures = PerMemoryType<std::size_t>{}]() mutable {
                for (auto const mem_type : ADDRESSABLE_MEMORY_TYPES) {
                    periodic_spill(mem_type, failures[mem_type]);
                }
            },
            *periodic_spill_check,
            // Only reached if the loop itself fails, e.g. initializing its CUDA context,
            // since `periodic_spill()` doesn't throw. Spilling on demand is unaffected.
            [](std::exception_ptr error) {
                std::cerr << "RapidsMPF warning: the periodic spill thread failed and "
                             "stopped, spilling continues on demand only: "
                          << detail::exception_message(error) << std::endl;
            }
        );
    }
}

void SpillManager::periodic_spill(MemoryType mem_type, std::size_t& failures) noexcept {
    try {
        std::ignore = spill_to_make_headroom(0, mem_type);
    } catch (...) {
        // A spill function threw, e.g. because a buffer could not be allocated in the
        // memory type it spills to. The spill functions leave data they could not move
        // where it is (or, for streaming messages, report the loss on extraction), so
        // keep going: the next periodic or on-demand spill retries. Report the first
        // failure of a streak, not every iteration.
        if (failures++ == 0) {
            std::cerr << "RapidsMPF warning: periodic spill from " << to_string(mem_type)
                      << " failed, will retry: "
                      << detail::exception_message(std::current_exception()) << std::endl;
        }
        return;
    }
    if (failures > 0) {
        std::cerr << "RapidsMPF warning: periodic spill from " << to_string(mem_type)
                  << " no longer fails, after " << failures << " failed attempt(s)"
                  << std::endl;
        failures = 0;
    }
}

SpillManager::~SpillManager() {
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->stop();
    }
}

SpillManager::SpillFunctionID SpillManager::add_spill_function(
    SpillFunction spill_function, int priority
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const id = spill_function_id_counter_++;
    RAPIDSMPF_EXPECTS(
        spill_functions_.insert({id, std::move(spill_function)}).second,
        "corrupted id counter"
    );
    spill_function_priorities_.insert({priority, id});

    // Make sure the spill thread is running.
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->resume();
    }
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

    // Asynchronously pause the spill thread if no spill functions are left.
    if (periodic_spill_thread_.has_value() && spill_functions_.empty()) {
        periodic_spill_thread_->pause_nb();
    }
}

std::size_t SpillManager::spill_unsafe(std::size_t amount, MemoryType mem_type) {
    if (amount == 0) {
        return 0;
    }

    // Marks a spill as in flight for the duration of this call, even if a spill
    // function throws. On exit, the generation is bumped before the in-flight count
    // drops, so an observer that sees no spill in flight also sees the new generation.
    struct InFlightGuard {
        SpillManager* self;
        MemoryType mem_type;

        ~InFlightGuard() {
            self->spill_generation_[mem_type].fetch_add(1, std::memory_order_release);
            self->spills_in_flight_[mem_type].fetch_sub(1, std::memory_order_release);
        }
    };

    spills_in_flight_[mem_type].fetch_add(1, std::memory_order_release);
    InFlightGuard const in_flight{.self = this, .mem_type = mem_type};

    std::size_t spilled{0};
    for (auto const [_, fid] : spill_function_priorities_) {
        if (spilled >= amount) {
            break;
        }
        spilled += spill_functions_.at(fid)(amount - spilled, mem_type);
    }
    return spilled;
}

bool SpillManager::spilling_now(MemoryType mem_type) const noexcept {
    return spills_in_flight_[mem_type].load(std::memory_order_acquire) > 0;
}

std::uint64_t SpillManager::spill_generation(MemoryType mem_type) const noexcept {
    return spill_generation_[mem_type].load(std::memory_order_acquire);
}

std::size_t SpillManager::spill(std::size_t amount, MemoryType mem_type) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    expect_spillable(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    return spill_unsafe(amount, mem_type);
}

SpillManager::HeadroomResult SpillManager::spill_to_make_headroom(
    std::int64_t headroom, MemoryType mem_type
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    expect_spillable(mem_type);
    std::lock_guard<std::mutex> lock(mutex_);
    std::int64_t const available = br_->memory_available_for_reservation(mem_type);
    if (headroom <= available) {
        return {.deficit = 0, .spilled = 0};
    }
    auto const deficit = safe_cast<std::size_t>(headroom - available);
    return {.deficit = deficit, .spilled = spill_unsafe(deficit, mem_type)};
}

}  // namespace rapidsmpf
