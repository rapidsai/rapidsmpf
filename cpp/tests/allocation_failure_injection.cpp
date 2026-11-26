/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allocation_failure_injection.hpp"

#include <cstdlib>
#include <new>

namespace {

// Plain thread-locals, so they can be used from `operator new` at any time.
thread_local bool injection_armed = false;
thread_local bool injection_fired = false;
thread_local std::size_t injection_countdown = 0;

}  // namespace

// Replaces the global `operator new` of the test binaries. The other forms (`new[]`,
// nothrow) are implemented in terms of this one by the standard library. The aligned
// forms are not replaced and use their own allocation functions.
void* operator new(std::size_t size) {
    if (injection_armed) {
        if (injection_countdown == 0) {
            injection_armed = false;
            injection_fired = true;
            throw std::bad_alloc{};
        }
        --injection_countdown;
    }
    if (void* ptr = std::malloc(size == 0 ? 1 : size)) {
        return ptr;
    }
    throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

namespace rapidsmpf::test {

ScopedAllocationFailure::ScopedAllocationFailure(std::size_t n) noexcept {
    injection_countdown = n;
    injection_fired = false;
    injection_armed = true;
}

ScopedAllocationFailure::~ScopedAllocationFailure() noexcept {
    injection_armed = false;
}

bool ScopedAllocationFailure::fired() const noexcept {
    return injection_fired;
}

}  // namespace rapidsmpf::test
