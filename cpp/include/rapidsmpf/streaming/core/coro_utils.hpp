/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Collect the results of multiple finished coroutines.
 *
 * This helper consumes a range of coroutine result objects (e.g., from
 * `coro::when_all` or `coro::when_any`) and extracts their return values by
 * invoking `.return_value()` on each element.
 *
 * - If the tasks produce a non-void type `T`, all values are collected into a
 *   `std::vector<T>` and returned.
 * - If the tasks return `void`, the function simply invokes `.return_value()`
 *   on each element to surface any unhandled exceptions and then returns `void`.
 *
 * @tparam Range A range type whose elements support a `.return_value()` member
 * function. Typically the result of functions and coroutines like `coro::when_all`
 * and `coro::wait_all`
 *
 * @param task_results A range of completed coroutine results.
 * @return `std::vector<T>` if the underlying tasks return a value of type `T` or
 * `void` if the underlying tasks return `void`.
 *
 * @note All result types must be the same. If your coroutines produce heterogeneous
 * result types, this helper cannot be used; you must instead extract each result
 * manually by calling `.return_value()` on each element.
 *
 * @note The return values of libcoro's gather functions such as `coro::when_all`
 * and `coro::wait_all` must always be retrieved by calling `.return_value()` (either
 * directly or via this helper). Failing to do so leaves exceptions unobserved, which
 * can cause the streaming pipeline to deadlock or hang indefinitely while waiting for
 * error propagation.
 */
template <std::ranges::range Range>
auto coro_results(Range&& task_results) {
    using first_ref_t = std::ranges::range_value_t<Range>;
    using raw_ret_t = decltype(std::declval<first_ref_t>().return_value());
    using val_t = std::remove_cvref_t<raw_ret_t>;

    if constexpr (std::is_void_v<val_t>) {
        // Just surface exceptions; return void.
        for (auto&& r : task_results) {
            r.return_value();
        }
    } else {
        std::vector<val_t> ret;
        if constexpr (std::ranges::sized_range<Range>) {
            ret.reserve(task_results.size());
        }
        for (auto&& r : task_results) {
            ret.emplace_back(r.return_value());
        }
        return ret;
    }
}

}  // namespace rapidsmpf::streaming
