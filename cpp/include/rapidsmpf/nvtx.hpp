/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <limits>
#include <stdexcept>
#include <type_traits>

#include <nvtx3/nvtx3.hpp>

/**
 * @brief Help function to convert value to 64 bit signed integer
 */
template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
[[nodiscard]] std::int64_t convert_to_64bit(T value) {
    if constexpr (std::numeric_limits<T>::max() > std::numeric_limits<std::int64_t>::max())
    {
        if (value > std::numeric_limits<std::int64_t>::max()) {
            throw std::overflow_error(
                "convert_to_64bit(x): x too large to fit std::int64_t"
            );
        }
    }
    return std::int64_t(value);
}

/**
 * @brief Help function to convert value to 64 bit float
 */
template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
[[nodiscard]] double convert_to_64bit(T value) {
    return double(value);
}

/**
 * @brief Tag type for rapidsmpf's NVTX domain.
 */
struct rapidsmpf_domain {
    static constexpr char const* name{"rapidsmpf"};  ///< nvtx domain name
};

// Macro to concatenate two tokens x and y.
#define RAPIDSMPF_CONCAT_HELPER(x, y) x##y
#define RAPIDSMPF_CONCAT(x, y) RAPIDSMPF_CONCAT_HELPER(x, y)

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define RAPIDSMPF_REGISTER_STRING(msg)                                         \
    [](const char* a_msg) -> auto& {                                           \
        static nvtx3::registered_string_in<rapidsmpf_domain> a_reg_str{a_msg}; \
        return a_reg_str;                                                      \
    }(msg)

// Macro overloads of RAPIDSMPF_NVTX_FUNC_RANGE
#define RAPIDSMPF_NVTX_FUNC_RANGE_IMPL() NVTX3_FUNC_RANGE_IN(rapidsmpf_domain)

#define RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL(msg, val)             \
    nvtx3::scoped_range_in<rapidsmpf_domain> RAPIDSMPF_CONCAT( \
        _rapidsmpf_nvtx_range, __LINE__                        \
    ) {                                                        \
        nvtx3::event_attributes {                              \
            RAPIDSMPF_REGISTER_STRING(msg), nvtx3::payload {   \
                convert_to_64bit(val)                          \
            }                                                  \
        }                                                      \
    }

#define RAPIDSMPF_NVTX_MARKER_IMPL(msg, val)                                  \
    nvtx3::mark_in<rapidsmpf_domain>(nvtx3::event_attributes{                 \
        RAPIDSMPF_REGISTER_STRING(msg), nvtx3::payload{convert_to_64bit(val)} \
    })

/**
 * @brief Convenience macro for generating an NVTX range in the `rapidsmpf` domain
 * from the lifetime of a function.
 *
 * Takes no argument. The name of the immediately enclosing function returned by
 * `__func__` is used as the message.
 *
 * Example:
 * ```
 * void some_function(){
 *    RAPIDSMPF_NVTX_FUNC_RANGE();  // The name `some_function` is used as the message
 *    ...
 * }
 * ```
 */
#define RAPIDSMPF_NVTX_FUNC_RANGE() RAPIDSMPF_NVTX_FUNC_RANGE_IMPL()

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `rapidsmpf` domain
 * to annotate a time duration.
 *
 * Takes two arguments (message, payload).
 *
 * Example:
 * ```
 * void some_function(){
 *    RAPIDSMPF_NVTX_SCOPED_RANGE("my function", 42);
 *    ...
 * }
 * ```
 */
#define RAPIDSMPF_NVTX_SCOPED_RANGE(msg, val) RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL(msg, val)

/**
 * @brief Convenience macro for generating an NVTX marker in the `rapidsmpf` domain to
 * annotate a certain time point.
 *
 * Takes two arguments (message, payload). Use this macro to annotate asynchronous
 * operations.
 */
#define RAPIDSMPF_NVTX_MARKER(message, payload) \
    RAPIDSMPF_NVTX_MARKER_IMPL(message, payload)
