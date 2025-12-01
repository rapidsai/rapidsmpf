/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf {

namespace {
/**
 * @brief Retrieves a value from a statistics array or accumulates the total.
 *
 * @param arr        The array containing statistics for each allocator type.
 * @param alloc_type The type of allocator to retrieve data for. If `AllocType::ALL`,
 *                   the function returns the sum across all entries in the array.
 * @return The requested statistic value or the accumulated total.
 */
std::int64_t get_or_accumulate(
    ScopedMemoryRecord::AllocTypeArray const& arr,
    ScopedMemoryRecord::AllocType alloc_type
) noexcept {
    if (alloc_type == ScopedMemoryRecord::AllocType::ALL) {
        return std::accumulate(arr.begin(), arr.end(), std::int64_t{0});
    }
    return arr[static_cast<std::size_t>(alloc_type)];
}
}  // namespace

std::int64_t ScopedMemoryRecord::num_total_allocs(AllocType alloc_type) const noexcept {
    return get_or_accumulate(num_total_allocs_, alloc_type);
}

std::int64_t ScopedMemoryRecord::num_current_allocs(AllocType alloc_type) const noexcept {
    return get_or_accumulate(num_current_allocs_, alloc_type);
}

std::int64_t ScopedMemoryRecord::current(AllocType alloc_type) const noexcept {
    return get_or_accumulate(current_, alloc_type);
}

std::int64_t ScopedMemoryRecord::total(AllocType alloc_type) const noexcept {
    return get_or_accumulate(total_, alloc_type);
}

std::int64_t ScopedMemoryRecord::peak(AllocType alloc_type) const noexcept {
    if (alloc_type == AllocType::ALL) {
        return highest_peak_;
    }
    return peak_[static_cast<std::size_t>(alloc_type)];
}

void ScopedMemoryRecord::record_allocation(AllocType alloc_type, std::int64_t nbytes) {
    RAPIDSMPF_EXPECTS(
        alloc_type != AllocType::ALL,
        "AllocType::ALL may not be used to record allocation"
    );
    auto at = static_cast<std::size_t>(alloc_type);
    ++num_total_allocs_[at];
    ++num_current_allocs_[at];
    current_[at] += nbytes;
    total_[at] += nbytes;
    peak_[at] = std::max(peak_[at], current_[at]);
    highest_peak_ = std::max(highest_peak_, current());
}

void ScopedMemoryRecord::record_deallocation(AllocType alloc_type, std::int64_t nbytes) {
    RAPIDSMPF_EXPECTS(
        alloc_type != AllocType::ALL,
        "AllocType::ALL may not be used to record deallocation"
    );
    auto at = static_cast<std::size_t>(alloc_type);
    current_[at] -= nbytes;
    --num_current_allocs_[at];
}

ScopedMemoryRecord& ScopedMemoryRecord::add_subscope(ScopedMemoryRecord const& subscope) {
    highest_peak_ = std::max(highest_peak_, current() + subscope.highest_peak_);
    for (AllocType type : {AllocType::PRIMARY, AllocType::FALLBACK}) {
        auto i = static_cast<std::size_t>(type);
        peak_[i] = std::max(peak_[i], current_[i] + subscope.peak_[i]);
        num_total_allocs_[i] += subscope.num_total_allocs_[i];
        num_current_allocs_[i] += subscope.num_current_allocs_[i];
        current_[i] += subscope.current_[i];
        total_[i] += subscope.total_[i];
    }
    return *this;
}

ScopedMemoryRecord& ScopedMemoryRecord::add_scope(ScopedMemoryRecord const& scope) {
    highest_peak_ = std::max(highest_peak_, scope.highest_peak_);
    for (AllocType type : {AllocType::PRIMARY, AllocType::FALLBACK}) {
        auto i = static_cast<std::size_t>(type);
        peak_[i] = std::max(peak_[i], scope.peak_[i]);
        current_[i] += scope.current_[i];
        total_[i] += scope.total_[i];
        num_total_allocs_[i] += scope.num_total_allocs_[i];
        num_current_allocs_[i] += scope.num_current_allocs_[i];
    }
    return *this;
}

}  // namespace rapidsmpf
