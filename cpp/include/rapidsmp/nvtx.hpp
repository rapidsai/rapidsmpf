/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * @brief Tag type for rapidsmp's NVTX domain.
 */
struct rapidsmp_domain {
    static constexpr char const* name{"rapidsmp"};  ///< nvtx domain name
};

// Macro to concatenate two tokens x and y.
#define KVIKIO_CONCAT_HELPER(x, y) x##y
#define KVIKIO_CONCAT(x, y) KVIKIO_CONCAT_HELPER(x, y)

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define KVIKIO_REGISTER_STRING(msg)                                           \
    [](const char* a_msg) -> auto& {                                          \
        static nvtx3::registered_string_in<rapidsmp_domain> a_reg_str{a_msg}; \
        return a_reg_str;                                                     \
    }(msg)

// Macro overloads of KVIKIO_NVTX_FUNC_RANGE
#define KVIKIO_NVTX_FUNC_RANGE_IMPL() NVTX3_FUNC_RANGE_IN(rapidsmp_domain)

#define KVIKIO_NVTX_SCOPED_RANGE_IMPL(msg, val)            \
    nvtx3::scoped_range_in<rapidsmp_domain> KVIKIO_CONCAT( \
        _kvikio_nvtx_range, __LINE__                       \
    ) {                                                    \
        nvtx3::event_attributes {                          \
            KVIKIO_REGISTER_STRING(msg), nvtx3::payload {  \
                convert_to_64bit(val)                      \
            }                                              \
        }                                                  \
    }

#define KVIKIO_NVTX_MARKER_IMPL(msg, val)                                  \
    nvtx3::mark_in<rapidsmp_domain>(nvtx3::event_attributes{               \
        KVIKIO_REGISTER_STRING(msg), nvtx3::payload{convert_to_64bit(val)} \
    })

/**
 * @brief Convenience macro for generating an NVTX range in the `rapidsmp` domain
 * from the lifetime of a function.
 *
 * Takes no argument. The name of the immediately enclosing function returned by
 * `__func__` is used as the message.
 *
 * Example:
 * ```
 * void some_function(){
 *    KVIKIO_NVTX_FUNC_RANGE();  // The name `some_function` is used as the message
 *    ...
 * }
 * ```
 */
#define KVIKIO_NVTX_FUNC_RANGE() KVIKIO_NVTX_FUNC_RANGE_IMPL()

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `rapidsmp` domain
 * to annotate a time duration.
 *
 * Takes two arguments (message, payload).
 *
 * Example:
 * ```
 * void some_function(){
 *    KVIKIO_NVTX_SCOPED_RANGE("my function", 42);
 *    ...
 * }
 * ```
 */
#define KVIKIO_NVTX_SCOPED_RANGE(msg, val) KVIKIO_NVTX_SCOPED_RANGE_IMPL(msg, val)

/**
 * @brief Convenience macro for generating an NVTX marker in the `rapidsmp` domain to
 * annotate a certain time point.
 *
 * Takes two arguments (message, payload). Use this macro to annotate asynchronous I/O
 * operations, where the payload refers to the I/O size.
 *
 * Example:
 * ```
 * std::future<void> some_function(){
 *     size_t io_size{2077};
 *     KVIKIO_NVTX_MARKER("I/O operation", io_size);
 *     perform_async_io_operation(io_size);
 *     ...
 * }
 * ```
 */
#define KVIKIO_NVTX_MARKER(message, payload) KVIKIO_NVTX_MARKER_IMPL(message, payload)
