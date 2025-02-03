/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <stdexcept>  // NOLINT(unused-includes)

namespace rapidsmp {

// Stringify a macro argument
#define RAPIDSMP_STRINGIFY_DETAIL(x) #x
#define RAPIDSMP_STRINGIFY(x) RAPIDSMP_STRINGIFY_DETAIL(x)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing `std::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws std::logic_error
 * RAPIDSMP_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * RAPIDSMP_EXPECTS(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to `std::logic_error`.
 *
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define RAPIDSMP_EXPECTS(...)                                                       \
    GET_RAPIDSMP_EXPECTS_MACRO(__VA_ARGS__, RAPIDSMP_EXPECTS_3, RAPIDSMP_EXPECTS_2) \
    (__VA_ARGS__)

#define GET_RAPIDSMP_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define RAPIDSMP_EXPECTS_3(_condition, _reason, _exception_type)                    \
    do {                                                                            \
        static_assert(std::is_base_of_v<std::exception, _exception_type>);          \
        (_condition) ? static_cast<void>(0)                                         \
                     : throw _exception_type /*NOLINT(bugprone-macro-parentheses)*/ \
            {"RAPIDSMP failure at: " __FILE__                                       \
             ":" RAPIDSMP_STRINGIFY(__LINE__) ": " _reason};                        \
    } while (0)

#define RAPIDSMP_EXPECTS_2(_condition, _reason) \
    RAPIDSMP_EXPECTS_3(_condition, _reason, std::logic_error)


/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```c++
 * // Throws `std::logic_error`
 * RAPIDSMP_FAIL("Unsupported code path");
 *
 * // Throws `std::runtime_error`
 * RAPIDSMP_FAIL("Unsupported code path", std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to `std::logic_error`.
 *
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define RAPIDSMP_FAIL(...)                                                 \
    GET_RAPIDSMP_FAIL_MACRO(__VA_ARGS__, RAPIDSMP_FAIL_2, RAPIDSMP_FAIL_1) \
    (__VA_ARGS__)

#define GET_RAPIDSMP_FAIL_MACRO(_1, _2, NAME, ...) NAME

#define RAPIDSMP_FAIL_2(_what, _exception_type)                                     \
    /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                  \
    throw _exception_type {                                                         \
        "RAPIDSMP failure at:" __FILE__ ":" RAPIDSMP_STRINGIFY(__LINE__) ": " _what \
    }

#define RAPIDSMP_FAIL_1(_what) RAPIDSMP_FAIL_2(_what, std::logic_error)

}  // namespace rapidsmp
