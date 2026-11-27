/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

namespace rapidsmpf::test {

/**
 * @brief Make one call to the global `operator new` on the calling thread throw.
 *
 * While in scope, the @p n-th (0-based) call to the global `operator new` on the calling
 * thread throws `std::bad_alloc`. Only one call fails, and other threads are not
 * affected. Used to check that an operation is exception safe at every allocation it
 * makes, by sweeping @p n until the operation completes without the failure firing.
 *
 * The test binaries replace the global `operator new` to support this (see
 * `allocation_failure_injection.cpp`). It forwards to `std::malloc` when not armed.
 */
class ScopedAllocationFailure {
  public:
    /**
     * @brief Arm the failure.
     *
     * @param n The number of allocations to let through before the one that fails.
     */
    explicit ScopedAllocationFailure(std::size_t n) noexcept;

    /// @brief Disarm the failure, if it has not fired.
    ~ScopedAllocationFailure() noexcept;

    ScopedAllocationFailure(ScopedAllocationFailure const&) = delete;
    ScopedAllocationFailure& operator=(ScopedAllocationFailure const&) = delete;

    /**
     * @brief Whether the failure has fired.
     *
     * @return `true` if an allocation was made to fail.
     */
    [[nodiscard]] bool fired() const noexcept;
};

}  // namespace rapidsmpf::test
