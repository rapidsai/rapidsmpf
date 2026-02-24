/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {


SpillManager::SpillManager(
    BufferResource* br, std::optional<Duration> periodic_spill_check
)
    : br_{br} {
    if (periodic_spill_check.has_value()) {
        periodic_spill_thread_.emplace(
            [this]() { spill_to_make_headroom(0); }, *periodic_spill_check
        );
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

std::size_t SpillManager::spill(std::size_t amount) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::size_t spilled{0};
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto const [_, fid] : spill_function_priorities_) {
        if (spilled >= amount) {
            break;
        }
        spilled += spill_functions_.at(fid)(amount - spilled);
    }
    lock.unlock();
    auto& stats = *br_->statistics();
    if (spilled < amount) {
        // TODO: use a "max" statistic when it is available, for now we use the average.
        stats.register_formatter(
            "spill-manager-limit-breach",
            [](std::ostream& os, std::vector<rapidsmpf::Statistics::Stat> const& s) {
                os << "avg " << format_nbytes(s[0].value() / s[0].count());
            }
        );
        stats.add_stat("spill-manager-limit-breach", amount - spilled);
    }
    return spilled;
}

std::size_t SpillManager::spill_to_make_headroom(std::int64_t headroom) {
    // TODO: check other memory types.
    std::int64_t available = br_->memory_available(MemoryType::DEVICE)();
    if (headroom <= available) {
        return 0;
    }
    return spill(safe_cast<std::size_t>(headroom - available));
}

}  // namespace rapidsmpf
