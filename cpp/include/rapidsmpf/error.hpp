/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>
#include <iostream>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 * @ingroup errors
 */
struct cuda_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief Exception thrown when a RapidsMPF allocation fails.
 *
 * @ingroup errors
 */
class bad_alloc : public std::bad_alloc {
  public:
    /**
     * @brief Construct a bad_alloc with the error message.
     *
     * @param msg Message to be associated with the exception.
     */
    explicit bad_alloc(char const* msg)
        : what_{std::string{std::bad_alloc::what()} + ": " + msg} {}

    /**
     * @brief Construct a bad_alloc with the error message.
     *
     * @param msg Message to be associated with the exception.
     */
    explicit bad_alloc(std::string const& msg) : bad_alloc{msg.c_str()} {}

    /**
     * @brief Return the explanatory string.
     *
     * @return The explanatory string.
     */
    [[nodiscard]] char const* what() const noexcept override {
        return what_.c_str();
    }

  private:
    std::string what_;
};

/**
 * @brief Exception thrown when RapidsMPF runs out of memory.
 *
 * @ingroup errors
 *
 * This exception should only be thrown when we know a resource is out of memory.
 */
class out_of_memory : public bad_alloc {
  public:
    /**
     * @brief Construct an out_of_memory with the error message.
     *
     * @param msg Message to be associated with the exception.
     */
    explicit out_of_memory(char const* msg)
        : bad_alloc{std::string{"out_of_memory: "} + msg} {}

    /**
     * @brief Construct an out_of_memory with the error message.
     *
     * @param msg Message to be associated with the exception.
     */
    explicit out_of_memory(std::string const& msg) : out_of_memory{msg.c_str()} {}
};

/**
 * @brief Exception thrown when a memory reservation fails in RapidsMPF.
 *
 * @ingroup errors
 *
 * This exception is thrown when attempting to reserve memory fails, or when an
 * existing reservation is insufficient for a requested allocation. It does not
 * necessarily indicate that the system is out of physical memory, only that the
 * reservation contract could not be satisfied.
 */
class reservation_error : public bad_alloc {
  public:
    /**
     * @brief Construct a reservation_error with an error message.
     *
     * @param msg Message to be associated with the exception.
     */
    explicit reservation_error(char const* msg)
        : bad_alloc{std::string{"reservation_error: "} + msg} {}

    /**
     * @brief Construct a reservation_error with an error message.
     *
     * @param msg Message to be associated with the exception.
     */
    explicit reservation_error(std::string const& msg) : reservation_error{msg.c_str()} {}
};

namespace detail {

/**
 * @brief Build an error message with source location information.
 *
 * @param reason The error reason message.
 * @param loc The source location where the error occurred.
 * @return The formatted error message string.
 */
inline std::string build_error_message(
    std::string_view reason, std::source_location const& loc
) {
    std::ostringstream ss;
    ss << "RAPIDSMPF failure at: " << loc.file_name() << ":" << loc.line() << ": "
       << reason;
    return ss.str();
}

/**
 * @brief Core implementation for RAPIDSMPF_EXPECTS.
 *
 * @param condition The condition to check.
 * @param reason The error reason if condition is false.
 * @param loc The source location (automatically captured).
 * @param throw_fn Callable that throws the appropriate exception type.
 */
template <typename ThrowFn>
inline void expects_impl(
    bool condition,
    std::string_view reason,
    std::source_location const& loc,
    ThrowFn&& throw_fn
) {
    if (!condition) {
        throw_fn(build_error_message(reason, loc));
    }
}

/**
 * @brief Build a CUDA error message with source location information.
 *
 * @param error The CUDA error code.
 * @param loc The source location where the error occurred.
 * @return The formatted CUDA error message string.
 */
inline std::string build_cuda_error_message(
    cudaError_t error, std::source_location const& loc
) {
    std::ostringstream ss;
    ss << "CUDA error at: " << loc.file_name() << ":" << loc.line() << ": "
       << cudaGetErrorName(error) << " " << cudaGetErrorString(error);
    return ss.str();
}

/**
 * @brief Build a CUDA allocation error message with source location information.
 *
 * @param error The CUDA error code.
 * @param num_bytes Number of bytes that failed to allocate.
 * @param loc The source location where the error occurred.
 * @return The formatted CUDA allocation error message string.
 */
inline std::string build_cuda_alloc_error_message(
    cudaError_t error, std::size_t num_bytes, std::source_location const& loc
) {
    std::ostringstream ss;
    ss << "CUDA error (failed to allocate " << num_bytes
       << " bytes) at: " << loc.file_name() << ":" << loc.line() << ": "
       << cudaGetErrorName(error) << " " << cudaGetErrorString(error);
    return ss.str();
}

/**
 * @brief Print a fatal error message and terminate.
 *
 * Prints an error message to stderr and calls std::terminate(). Use this for
 * fatal errors in contexts where exceptions cannot be thrown, for example in
 * destructors.
 *
 * @param reason The error reason message.
 * @param loc The source location where the error occurred.
 */
[[noreturn]] inline void fatal_error(
    std::string_view reason, std::source_location const& loc
) noexcept {
    std::cerr << "RAPIDSMPF FATAL ERROR at: " << loc.file_name() << ":" << loc.line()
              << ": " << reason << std::endl;
    std::terminate();
}

}  // namespace detail

/**
 * @brief Check a condition and terminate if false.
 *
 * This is the fatal (non-throwing) version of RAPIDSMPF_EXPECTS. It checks a
 * condition and, if false, prints an error message to stderr and calls
 * std::terminate(). Use this in contexts where exceptions cannot be thrown,
 * such as destructors, noexcept functions, or when recovery is impossible.
 *
 * @param condition The condition to check.
 * @param reason The error message if the condition is false.
 * @param loc The source location (automatically captured at call site).
 */
inline void RAPIDSMPF_EXPECTS_FATAL(
    bool condition,
    std::string_view reason,
    std::source_location loc = std::source_location::current()
) noexcept {
    if (!condition) {
        detail::fatal_error(reason, loc);
    }
}

/**
 * @brief Indicate a fatal error and terminate the program.
 *
 * Prints an error message to stderr and calls std::terminate(). Use this for
 * fatal errors in contexts where exceptions cannot be thrown, such as
 * destructors, noexcept functions, or when recovery is impossible.
 *
 * @param reason The error message describing the fatal error.
 * @param loc The source location (automatically captured at call site).
 */
[[noreturn]] inline void RAPIDSMPF_FATAL(
    std::string_view reason, std::source_location loc = std::source_location::current()
) noexcept {
    detail::fatal_error(reason, loc);
}

/**
 * @brief Check a condition and throw an exception if it is violated.
 *
 * Defaults to throwing `std::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * @code{.cpp}
 * // Throws std::logic_error
 * RAPIDSMPF_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // Throws std::runtime_error
 * RAPIDSMPF_EXPECTS(p != nullptr, "Unexpected null pointer", std::runtime_error);
 * @endcode
 *
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or false,
 *     and is the condition being checked.
 *   - The second argument is a string used to construct the `what` of the exception.
 *   - When given, the third argument is the exception type to be thrown. When not
 *     specified, it defaults to `std::logic_error`.
 *
 * @throws _exception_type If the condition evaluates to false.
 */
#define RAPIDSMPF_EXPECTS(...)                                                         \
    GET_RAPIDSMPF_EXPECTS_MACRO(__VA_ARGS__, RAPIDSMPF_EXPECTS_3, RAPIDSMPF_EXPECTS_2) \
    (__VA_ARGS__)

#define GET_RAPIDSMPF_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define RAPIDSMPF_EXPECTS_3(_condition, _reason, _exception_type)          \
    do {                                                                   \
        static_assert(std::is_base_of_v<std::exception, _exception_type>); \
        rapidsmpf::detail::expects_impl(                                   \
            static_cast<bool>(_condition),                                 \
            (_reason),                                                     \
            std::source_location::current(),                               \
            [](auto&& msg) { throw _exception_type{msg}; }                 \
        );                                                                 \
    } while (0)

#define RAPIDSMPF_EXPECTS_2(_condition, _reason) \
    RAPIDSMPF_EXPECTS_3(_condition, _reason, std::logic_error)

/**
 * @brief Indicate that an erroneous code path has been taken.
 *
 * Example usage:
 * @code{.cpp}
 * // Throws std::logic_error
 * RAPIDSMPF_FAIL("Unsupported code path");
 *
 * // Throws std::runtime_error
 * RAPIDSMPF_FAIL("Unsupported code path", std::runtime_error);
 * @endcode
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument is a string used to construct the `what` of the exception.
 *   - When given, the second argument is the exception type to be thrown. When not
 *     specified, it defaults to `std::logic_error`.
 *
 * @throws _exception_type Always throws.
 */
#define RAPIDSMPF_FAIL(...)                                                   \
    GET_RAPIDSMPF_FAIL_MACRO(__VA_ARGS__, RAPIDSMPF_FAIL_2, RAPIDSMPF_FAIL_1) \
    (__VA_ARGS__)

#define GET_RAPIDSMPF_FAIL_MACRO(_1, _2, NAME, ...) NAME

#define RAPIDSMPF_FAIL_2(what_, _exception_type)                                         \
    throw _exception_type {                                                              \
        rapidsmpf::detail::build_error_message((what_), std::source_location::current()) \
    }

#define RAPIDSMPF_FAIL_1(what_) RAPIDSMPF_FAIL_2(what_, std::logic_error)

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, it calls `cudaGetLastError()` to clear the error and throws an
 * exception describing the CUDA error.
 *
 * Defaults to throwing `rapidsmpf::cuda_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * @code{.cpp}
 * // Throws rapidsmpf::cuda_error if cudaMalloc fails
 * RAPIDSMPF_CUDA_TRY(cudaMalloc(&p, 100));
 *
 * // Throws std::runtime_error if cudaMalloc fails
 * RAPIDSMPF_CUDA_TRY(cudaMalloc(&p, 100), std::runtime_error);
 * @endcode
 */
#define RAPIDSMPF_CUDA_TRY(...)                                 \
    GET_RAPIDSMPF_CUDA_TRY_MACRO(                               \
        __VA_ARGS__, RAPIDSMPF_CUDA_TRY_2, RAPIDSMPF_CUDA_TRY_1 \
    )                                                           \
    (__VA_ARGS__)
#define GET_RAPIDSMPF_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define RAPIDSMPF_CUDA_TRY_2(_call, _exception_type)                       \
    do {                                                                   \
        cudaError_t const error = (_call);                                 \
        if (cudaSuccess != error) {                                        \
            cudaGetLastError();                                            \
            throw _exception_type /*NOLINT(bugprone-macro-parentheses)*/ { \
                ::rapidsmpf::detail::build_cuda_error_message(             \
                    error, std::source_location::current()                 \
                )                                                          \
            };                                                             \
        }                                                                  \
    } while (0)
#define RAPIDSMPF_CUDA_TRY_1(_call) RAPIDSMPF_CUDA_TRY_2(_call, rapidsmpf::cuda_error)

/**
 * @brief Error checking macro for CUDA runtime API calls that terminates on error.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, it calls `cudaGetLastError()` to clear the error and terminates
 * with a fatal error message describing the CUDA error.
 *
 * Use this in contexts where exceptions cannot be thrown, such as destructors,
 * noexcept functions, or when recovery is impossible.
 *
 * Example usage:
 * @code{.cpp}
 * RAPIDSMPF_CUDA_TRY_FATAL(cudaDeviceSynchronize());
 * @endcode
 */
#define RAPIDSMPF_CUDA_TRY_FATAL(_call)                        \
    do {                                                       \
        cudaError_t const error = (_call);                     \
        if (cudaSuccess != error) {                            \
            cudaGetLastError();                                \
            ::rapidsmpf::detail::fatal_error(                  \
                ::rapidsmpf::detail::build_cuda_error_message( \
                    error, std::source_location::current()     \
                ),                                             \
                std::source_location::current()                \
            );                                                 \
        }                                                      \
    } while (0)

/**
 * @brief Error checking macro for CUDA memory allocation calls.
 *
 * Invokes a CUDA memory allocation function call. If the call does not return
 * `cudaSuccess`, it calls `cudaGetLastError()` to clear the error and throws an
 * exception describing the CUDA error.
 *
 * Defaults to throwing `rapidsmpf::bad_alloc`, but when `cudaErrorMemoryAllocation`
 * is returned, `rapidsmpf::out_of_memory` is thrown instead.
 *
 * This macro can be called with either one or two arguments:
 * - RAPIDSMPF_CUDA_TRY_ALLOC(cuda_call): Performs error checking without specifying
 * bytes.
 * - RAPIDSMPF_CUDA_TRY_ALLOC(cuda_call, num_bytes): Includes the byte count in the error
 * message.
 */
#define RAPIDSMPF_CUDA_TRY_ALLOC(...)                                       \
    GET_RAPIDSMPF_CUDA_TRY_ALLOC_MACRO(                                     \
        __VA_ARGS__, RAPIDSMPF_CUDA_TRY_ALLOC_2, RAPIDSMPF_CUDA_TRY_ALLOC_1 \
    )                                                                       \
    (__VA_ARGS__)
#define GET_RAPIDSMPF_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME

#define RAPIDSMPF_CUDA_TRY_ALLOC_2(_call, num_bytes)                              \
    do {                                                                          \
        cudaError_t const error = (_call);                                        \
        if (cudaSuccess != error) {                                               \
            cudaGetLastError();                                                   \
            auto const msg = ::rapidsmpf::detail::build_cuda_alloc_error_message( \
                error, (num_bytes), std::source_location::current()               \
            );                                                                    \
            if (cudaErrorMemoryAllocation == error) {                             \
                throw rapidsmpf::out_of_memory{msg};                              \
            }                                                                     \
            throw rapidsmpf::bad_alloc{msg};                                      \
        }                                                                         \
    } while (0)

#define RAPIDSMPF_CUDA_TRY_ALLOC_1(_call)                                   \
    do {                                                                    \
        cudaError_t const error = (_call);                                  \
        if (cudaSuccess != error) {                                         \
            cudaGetLastError();                                             \
            auto const msg = ::rapidsmpf::detail::build_cuda_error_message( \
                error, std::source_location::current()                      \
            );                                                              \
            if (cudaErrorMemoryAllocation == error) {                       \
                throw rapidsmpf::out_of_memory{msg};                        \
            }                                                               \
            throw rapidsmpf::bad_alloc{msg};                                \
        }                                                                   \
    } while (0)

}  // namespace rapidsmpf
