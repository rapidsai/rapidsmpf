/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

#include <rapidsmpf/memory/scoped_memory_record.hpp>

namespace rapidsmpf {

std::int64_t ScopedMemoryRecord::num_total_allocs() const noexcept {
    return num_total_allocs_;
}

std::int64_t ScopedMemoryRecord::num_current_allocs() const noexcept {
    return num_current_allocs_;
}

std::int64_t ScopedMemoryRecord::current() const noexcept {
    return current_;
}

std::int64_t ScopedMemoryRecord::total() const noexcept {
    return total_;
}

std::int64_t ScopedMemoryRecord::peak() const noexcept {
    return peak_;
}

std::int64_t ScopedMemoryRecord::max() const noexcept {
    return max_;
}

void ScopedMemoryRecord::record_allocation(std::int64_t nbytes) {
    ++num_total_allocs_;
    ++num_current_allocs_;
    current_ += nbytes;
    total_ += nbytes;
    peak_ = std::max(peak_, current_);
    max_ = std::max(max_, nbytes);
}

void ScopedMemoryRecord::record_deallocation(std::int64_t nbytes) {
    current_ -= nbytes;
    --num_current_allocs_;
}

ScopedMemoryRecord& ScopedMemoryRecord::add_subscope(ScopedMemoryRecord const& subscope) {
    peak_ = std::max(peak_, current_ + subscope.peak_);
    max_ = std::max(max_, subscope.max_);
    num_total_allocs_ += subscope.num_total_allocs_;
    num_current_allocs_ += subscope.num_current_allocs_;
    current_ += subscope.current_;
    total_ += subscope.total_;
    return *this;
}

ScopedMemoryRecord& ScopedMemoryRecord::add_scope(ScopedMemoryRecord const& scope) {
    peak_ = std::max(peak_, scope.peak_);
    max_ = std::max(max_, scope.max_);
    current_ += scope.current_;
    total_ += scope.total_;
    num_total_allocs_ += scope.num_total_allocs_;
    num_current_allocs_ += scope.num_current_allocs_;
    return *this;
}

}  // namespace rapidsmpf
