/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <nvtx3/nvtx3.hpp>

#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::detail {

/**
 * @brief Convert an integral value to a 64-bit signed integer.
 *
 * @tparam T An integral type.
 * @param value The value to convert.
 * @return The value as `std::int64_t`.
 */
template <typename T>
    requires std::is_integral_v<T>
[[nodiscard]] std::int64_t convert_to_64bit(T value) {
    if constexpr (std::numeric_limits<T>::max()
                  > std::numeric_limits<std::int64_t>::max())
    {
        if (value > std::numeric_limits<std::int64_t>::max()) {
            throw std::overflow_error(
                "convert_to_64bit(x): x too large to fit std::int64_t"
            );
        }
    }
    return rapidsmpf::safe_cast<std::int64_t>(value);
}

/**
 * @brief Convert a floating-point value to `double`.
 *
 * @tparam T A floating-point type.
 * @param value The value to convert.
 * @return The value as `double`.
 */
template <typename T>
    requires std::is_floating_point_v<T>
[[nodiscard]] double convert_to_64bit(T value) {
    return double(value);
}

/**
 * @brief Extract the qualified function name from a `__PRETTY_FUNCTION__` string.
 *
 * Strips the leading return-type token and the trailing parameter list so that
 * the result is just the qualified name, e.g.:
 *   - `"void rapidsmpf::Foo::bar(int)"` → `"rapidsmpf::Foo::bar"`
 *   - `"void rapidsmpf::baz()"`          → `"rapidsmpf::baz"`
 *   - `"Foo::Foo(int)"`                  → `"Foo::Foo"` (constructor, no return type)
 *
 * Works for class methods, constructors/destructors, and free functions alike.
 *
 * @param pretty The `std::source_location::function_name()` string of the enclosing
 *               function (equivalently, `__PRETTY_FUNCTION__` on GCC/Clang).
 * @return A `std::string_view` into @p pretty covering only the qualified name.
 */
[[nodiscard]] constexpr std::string_view extract_func_name(
    std::string_view pretty
) noexcept {
    // 1. Find the end boundary (either '(' or the end of the string)
    auto const paren = pretty.find('(');
    auto const end_pos = (paren == std::string_view::npos) ? pretty.size() : paren;

    // 2. Look for the last space before that boundary
    auto const space = pretty.rfind(' ', end_pos);

    // 3. If no space is found, the name starts at 0.
    //    Otherwise, start right after the space.
    auto const start_pos = (space == std::string_view::npos) ? 0 : space + 1;

    return pretty.substr(start_pos, end_pos - start_pos);
}

}  // namespace rapidsmpf::detail

/**
 * @brief Tag type for rapidsmpf's NVTX domain.
 */
struct rapidsmpf_domain {
    static constexpr char const* name{"rapidsmpf"};  ///< nvtx domain name
};

// Macro to create a static, registered string that will not have a name conflict with any
// registered string defined in the same scope.
#define RAPIDSMPF_REGISTER_STRING(msg)                                         \
    [](char const* a_msg) -> auto& {                                           \
        static nvtx3::registered_string_in<rapidsmpf_domain> a_reg_str{a_msg}; \
        return a_reg_str;                                                      \
    }(msg)

// Macro to create a static, registered string for the enclosing function.
//
// Uses std::source_location::current() to obtain the fully-qualified function
// name (return type + class::method + parameter list) and then strips the
// return type and parameter list, leaving e.g. "MyClass::my_method" for member
// functions and "my_function" for free functions.
//
// std::source_location::current() is evaluated as the argument to the immediately
// invoked lambda so it captures the call-site (the enclosing function) rather than
// the lambda body.  The extracted name is cached in a static std::string on first
// use so that nvtx3::registered_string_in receives a stable null-terminated pointer.
#define RAPIDSMPF_REGISTER_FUNC_STRING                                  \
    [](char const* pretty_fn) -> auto& {                                \
        static const std::string s_func_name{                           \
            rapidsmpf::detail::extract_func_name(pretty_fn)             \
        };                                                              \
        static nvtx3::registered_string_in<rapidsmpf_domain> a_reg_str{ \
            s_func_name.c_str()                                         \
        };                                                              \
        return a_reg_str;                                               \
    }(std::source_location::current().function_name())

// implement the func range macro with a value
#define RAPIDSMPF_NVTX_FUNC_RANGE_IMPL_WITH_VAL(val)           \
    static_assert(                                             \
        std::is_arithmetic_v<decltype(val)>,                   \
        "Value must be integral or floating point type"        \
    );                                                         \
    nvtx3::scoped_range_in<rapidsmpf_domain> RAPIDSMPF_CONCAT( \
        _rapidsmpf_nvtx_range, __LINE__                        \
    ) {                                                        \
        nvtx3::event_attributes {                              \
            RAPIDSMPF_REGISTER_FUNC_STRING, nvtx3::payload {   \
                rapidsmpf::detail::convert_to_64bit(val)       \
            }                                                  \
        }                                                      \
    }

// implement the func range macro without a value
#define RAPIDSMPF_NVTX_FUNC_RANGE_IMPL_WITHOUT_VAL()           \
    nvtx3::scoped_range_in<rapidsmpf_domain> RAPIDSMPF_CONCAT( \
        _rapidsmpf_nvtx_range, __LINE__                        \
    ) {                                                        \
        nvtx3::event_attributes {                              \
            RAPIDSMPF_REGISTER_FUNC_STRING                     \
        }                                                      \
    }

// Macro selector for 0 vs 1 arguments
#define RAPIDSMPF_GET_MACRO_FUNC(_0, _1, NAME, ...) NAME

// unwrap the arguments and call the appropriate macro
#define RAPIDSMPF_NVTX_FUNC_RANGE_IMPL(...)                                                                                                          \
    RAPIDSMPF_GET_MACRO_FUNC(dummy __VA_OPT__(, ) __VA_ARGS__, RAPIDSMPF_NVTX_FUNC_RANGE_IMPL_WITH_VAL, RAPIDSMPF_NVTX_FUNC_RANGE_IMPL_WITHOUT_VAL)( \
        __VA_ARGS__                                                                                                                                  \
    )

/**
 * @brief Convenience macro for generating an NVTX range in the `rapidsmpf` domain
 * from the lifetime of a function.
 *
 * The range label is derived from `__PRETTY_FUNCTION__` and includes the class name
 * for member functions (e.g. `"MyClass::my_method"`) while producing just the
 * function name for free functions (e.g. `"my_function"`).  Return types and
 * parameter lists are stripped so the label stays concise.
 *
 * Usage:
 * - `RAPIDSMPF_NVTX_FUNC_RANGE()` - Annotate with qualified function name only
 * - `RAPIDSMPF_NVTX_FUNC_RANGE(payload)` - Annotate with qualified function name and
 *   payload
 *
 * The optional argument is the payload to annotate (integral or floating-point value).
 *
 * Example:
 * ```cpp
 * // Free function → label "some_function"
 * void some_function() {
 *    RAPIDSMPF_NVTX_FUNC_RANGE();
 *    RAPIDSMPF_NVTX_FUNC_RANGE(42);  // With payload
 * }
 *
 * // Member function → label "MyClass::my_method"
 * void MyClass::my_method() {
 *    RAPIDSMPF_NVTX_FUNC_RANGE();
 * }
 * ```
 */
#define RAPIDSMPF_NVTX_FUNC_RANGE(...) RAPIDSMPF_NVTX_FUNC_RANGE_IMPL(__VA_ARGS__)

// implement the scoped range macro with a value
#define RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL_WITH_VAL(msg, val)    \
    nvtx3::scoped_range_in<rapidsmpf_domain> RAPIDSMPF_CONCAT( \
        _rapidsmpf_nvtx_range, __LINE__                        \
    ) {                                                        \
        nvtx3::event_attributes {                              \
            RAPIDSMPF_REGISTER_STRING(msg), nvtx3::payload {   \
                rapidsmpf::detail::convert_to_64bit(val)       \
            }                                                  \
        }                                                      \
    }

// implement the scoped range macro without a value
#define RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL_WITHOUT_VAL(msg)      \
    nvtx3::scoped_range_in<rapidsmpf_domain> RAPIDSMPF_CONCAT( \
        _rapidsmpf_nvtx_range, __LINE__                        \
    ) {                                                        \
        nvtx3::event_attributes {                              \
            RAPIDSMPF_REGISTER_STRING(msg)                     \
        }                                                      \
    }

// Macro to detect number of arguments (1 or 2)
#define RAPIDSMPF_GET_MACRO(_1, _2, NAME, ...) NAME

// unwrap the arguments and call the appropriate macro
#define RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL(...)        \
    RAPIDSMPF_GET_MACRO(                             \
        __VA_ARGS__,                                 \
        RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL_WITH_VAL,   \
        RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL_WITHOUT_VAL \
    )                                                \
    (__VA_ARGS__)

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `rapidsmpf` domain
 * to annotate a time duration.
 *
 * Usage:
 * - `RAPIDSMPF_NVTX_SCOPED_RANGE(message)` - Annotate with message only
 * - `RAPIDSMPF_NVTX_SCOPED_RANGE(message, payload)` - Annotate with message and payload
 *
 * The first argument is the message to annotate (const char*).
 * The second argument (optional) is the payload to annotate (integral or floating-point
 * value).
 *
 * Example:
 * ```
 * void some_function(){
 *    RAPIDSMPF_NVTX_SCOPED_RANGE("my function");        // Without payload
 *    RAPIDSMPF_NVTX_SCOPED_RANGE("my function", 42);    // With payload
 *    ...
 * }
 * ```
 */
#define RAPIDSMPF_NVTX_SCOPED_RANGE(...) RAPIDSMPF_NVTX_SCOPED_RANGE_IMPL(__VA_ARGS__)

/**
 * @brief Convenience macro for generating an NVTX scoped range in the `rapidsmpf` domain
 * that is only active when RAPIDSMPF_VERBOSE_INFO is defined.
 *
 * This macro behaves identically to RAPIDSMPF_NVTX_SCOPED_RANGE, but only creates
 * the NVTX range when the RAPIDSMPF_VERBOSE_INFO compile-time flag is set.
 *
 * Usage:
 * - `RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE(message)` - Annotate with message only
 * - `RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE(message, payload)` - Annotate with message and
 * payload
 *
 * Example:
 * ```
 * void some_function(){
 *    RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("detailed operation");
 *    RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("detailed operation", count);
 *    ...
 * }
 * ```
 */
#if RAPIDSMPF_VERBOSE_INFO
#define RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE(...) RAPIDSMPF_NVTX_SCOPED_RANGE(__VA_ARGS__)
#else
#define RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE(...)
#endif

#define RAPIDSMPF_NVTX_MARKER_IMPL(msg, val)                     \
    nvtx3::mark_in<rapidsmpf_domain>(nvtx3::event_attributes{    \
        RAPIDSMPF_REGISTER_STRING(msg),                          \
        nvtx3::payload{rapidsmpf::detail::convert_to_64bit(val)} \
    })

/**
 * @brief Convenience macro for generating an NVTX marker in the `rapidsmpf` domain to
 * annotate a certain time point.
 *
 * @param message The message to annotate.
 * @param payload The payload to annotate.
 *
 * Use this macro to annotate asynchronous operations.
 */
#define RAPIDSMPF_NVTX_MARKER(message, payload) \
    RAPIDSMPF_NVTX_MARKER_IMPL(message, payload)

/**
 * @brief Convenience macro for generating an NVTX marker in the `rapidsmpf` domain to
 * annotate a certain time point, that is only activate when RAPIDSMPF_VERBOSE_INFO is
 * defined.
 *
 * @param message The message to annotate.
 * @param payload The payload to annotate.
 *
 * Use this macro to annotate asynchronous operations.
 */
#if RAPIDSMPF_VERBOSE_INFO
#define RAPIDSMPF_NVTX_MARKER_VERBOSE(message, payload) \
    RAPIDSMPF_NVTX_MARKER_IMPL(message, payload)
#else
#define RAPIDSMPF_NVTX_MARKER_VERBOSE(message, payload)
#endif
