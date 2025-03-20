/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>  // NOLINT(unused-includes)

namespace rapidsmp {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 * @ingroup errors
 *
 */
struct cuda_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief Exception thrown when an RAPIDSMP allocation fails
 *
 * @ingroup errors
 *
 */
class bad_alloc : public std::bad_alloc {
  public:
    /**
     * @brief Constructs a bad_alloc with the error message.
     *
     * @param msg Message to be associated with the exception
     */
    bad_alloc(const char* msg)
        : _what{std::string{std::bad_alloc::what()} + ": " + msg} {}

    /**
     * @brief Constructs a bad_alloc with the error message.
     *
     * @param msg Message to be associated with the exception
     */
    bad_alloc(std::string const& msg) : bad_alloc{msg.c_str()} {}

    /**
     * @brief Returns the explanatory string.
     *
     * @return The explanatory string.
     */
    [[nodiscard]] const char* what() const noexcept override {
        return _what.c_str();
    }

  private:
    std::string _what;
};

/**
 * @brief Exception thrown when RAPIDSMP runs out of memory
 *
 * @ingroup errors
 *
 * This error should only be thrown when we know for sure a resource is out of memory.
 */
class out_of_memory : public bad_alloc {
  public:
    /**
     * @brief Constructs an out_of_memory with the error message.
     *
     * @param msg Message to be associated with the exception
     */
    out_of_memory(const char* msg) : bad_alloc{std::string{"out_of_memory: "} + msg} {}

    /**
     * @brief Constructs an out_of_memory with the error message.
     *
     * @param msg Message to be associated with the exception
     */
    out_of_memory(std::string const& msg) : out_of_memory{msg.c_str()} {}
};

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

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing rapidsmp::cuda_error, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```c++
 *
 * // Throws rapidsmp::cuda_error if `cudaMalloc` fails
 * RAPIDSMP_CUDA_TRY(cudaMalloc(&p, 100));
 *
 * // Throws std::runtime_error if `cudaMalloc` fails
 * RAPIDSMP_CUDA_TRY(cudaMalloc(&p, 100), std::runtime_error);
 * ```
 *
 */
#define RAPIDSMP_CUDA_TRY(...)                                                         \
    GET_RAPIDSMP_CUDA_TRY_MACRO(__VA_ARGS__, RAPIDSMP_CUDA_TRY_2, RAPIDSMP_CUDA_TRY_1) \
    (__VA_ARGS__)
#define GET_RAPIDSMP_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define RAPIDSMP_CUDA_TRY_2(_call, _exception_type)                                   \
    do {                                                                              \
        cudaError_t const error = (_call);                                            \
        if (cudaSuccess != error) {                                                   \
            cudaGetLastError();                                                       \
            /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                            \
            throw _exception_type{                                                    \
                std::string{"CUDA error at: "} + __FILE__ + ":"                       \
                + RAPIDSMP_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " \
                + cudaGetErrorString(error)                                           \
            };                                                                        \
        }                                                                             \
    } while (0)
#define RAPIDSMP_CUDA_TRY_1(_call) RAPIDSMP_CUDA_TRY_2(_call, rapidsmp::cuda_error)

/**
 * @brief Error checking macro for CUDA memory allocation calls.
 *
 * Invokes a CUDA memory allocation function call. If the call does not return
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 * Defaults to throwing rapidsmp::bad_alloc, but when `cudaErrorMemoryAllocation` is
 * returned, rapidsmp::out_of_memory is thrown instead.
 *
 * Can be called with either 1 or 2 arguments:
 * - RAPIDSMP_CUDA_TRY_ALLOC(cuda_call): Performs error checking without specifying bytes
 * - RAPIDSMP_CUDA_TRY_ALLOC(cuda_call, num_bytes): Includes the number of bytes in the
 * error message
 */
#define RAPIDSMP_CUDA_TRY_ALLOC(...)                                      \
    GET_RAPIDSMP_CUDA_TRY_ALLOC_MACRO(                                    \
        __VA_ARGS__, RAPIDSMP_CUDA_TRY_ALLOC_2, RAPIDSMP_CUDA_TRY_ALLOC_1 \
    )                                                                     \
    (__VA_ARGS__)
#define GET_RAPIDSMP_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME

#define RAPIDSMP_CUDA_TRY_ALLOC_2(_call, num_bytes)                                  \
    do {                                                                             \
        cudaError_t const error = (_call);                                           \
        if (cudaSuccess != error) {                                                  \
            cudaGetLastError();                                                      \
            auto const msg = std::string{"CUDA error (failed to allocate "}          \
                             + std::to_string(num_bytes) + " bytes) at: " + __FILE__ \
                             + ":" + RAPIDSMP_STRINGIFY(__LINE__) + ": "             \
                             + cudaGetErrorName(error) + " "                         \
                             + cudaGetErrorString(error);                            \
            if (cudaErrorMemoryAllocation == error) {                                \
                throw rapidsmp::out_of_memory{msg};                                  \
            }                                                                        \
            throw rapidsmp::bad_alloc{msg};                                          \
        }                                                                            \
    } while (0)

#define RAPIDSMP_CUDA_TRY_ALLOC_1(_call)                                     \
    do {                                                                     \
        cudaError_t const error = (_call);                                   \
        if (cudaSuccess != error) {                                          \
            cudaGetLastError();                                              \
            auto const msg = std::string{"CUDA error at: "} + __FILE__ + ":" \
                             + RAPIDSMP_STRINGIFY(__LINE__) + ": "           \
                             + cudaGetErrorName(error) + " "                 \
                             + cudaGetErrorString(error);                    \
            if (cudaErrorMemoryAllocation == error) {                        \
                throw rapidsmp::out_of_memory{msg};                          \
            }                                                                \
            throw rapidsmp::bad_alloc{msg};                                  \
        }                                                                    \
    } while (0)

}  // namespace rapidsmp
