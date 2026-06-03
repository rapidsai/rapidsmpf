/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

/**
 * @brief A logger base class for handling different levels of log messages.
 *
 * The logger class provides various logging methods with different verbosity levels.
 * It ensures thread-safety using a mutex and allows filtering of log messages
 * based on the configured verbosity level.
 *
 * The rank used in log message prefixes can either be supplied at construction
 * time or set later via `set_rank()`. The latter supports communicators (such as
 * UCXX) where the rank is only known after a bootstrap handshake.
 *
 * TODO: support writing to a file.
 */
class Logger : public std::enable_shared_from_this<Logger> {
  public:
    /**
     * @brief Log verbosity levels.
     *
     * Defines different logging levels for filtering messages.
     */
    enum class LOG_LEVEL : std::uint32_t {
        NONE = 0,  ///< No logging.
        PRINT,  ///< General print messages.
        WARN,  ///< Warning messages.
        INFO,  ///< Informational messages.
        DEBUG,  ///< Debug messages.
        TRACE  ///< Trace messages.
    };

    /**
     * @brief Log level names corresponding to the LOG_LEVEL enum.
     */
    static constexpr std::array<char const*, 6> LOG_LEVEL_NAMES{
        "NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"
    };

    /**
     * @brief Get the string name of a log level.
     *
     * @param level The log level.
     * @return The corresponding log level name or "UNKNOWN" if out of range.
     */
    static constexpr char const* level_name(LOG_LEVEL level) {
        auto index = static_cast<std::size_t>(level);
        return index < LOG_LEVEL_NAMES.size() ? LOG_LEVEL_NAMES[index] : "UNKNOWN";
    }

    /**
     * @brief Create a logger with a known rank.
     *
     * To control the verbosity level, set the configuration option "log" to
     * one of following:
     *  - NONE:  No logging.
     *  - PRINT: General print messages.
     *  - WARN:  Warning messages (default)
     *  - INFO:  Informational messages.
     *  - DEBUG: Debug messages.
     *  - TRACE: Trace messages.
     *
     * @param rank The rank of the calling process.
     * @param options Configuration options.
     * @return A shared pointer to the newly constructed logger.
     */
    [[nodiscard]] static std::shared_ptr<Logger> create(
        std::int32_t rank, config::Options options
    );

    /**
     * @brief Create a logger without a known rank.
     *
     * The rank defaults to `-1` and may be updated later via `set_rank()`.
     * This overload is intended for bootstrap scenarios where the rank is only
     * known after a network handshake but logging may already be required.
     *
     * @param options Configuration options.
     * @return A shared pointer to the newly constructed logger.
     */
    [[nodiscard]] static std::shared_ptr<Logger> create(config::Options options);

    virtual ~Logger() noexcept = default;

    // `Logger` is owned exclusively through `std::shared_ptr`. Use `create()`
    // to construct instances.
    Logger(Logger const&) = delete;
    Logger& operator=(Logger const&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

    /**
     * @brief Get the verbosity level of the logger.
     *
     * @return The verbosity level.
     */
    LOG_LEVEL verbosity_level() const {
        return level_;
    }

    /**
     * @brief Update the rank used in log message prefixes.
     *
     * Thread-safe (acquires the same mutex used to serialize log output). May
     * be called any number of times. Concurrent log calls will observe either
     * the old or the new value, never a partially written one.
     *
     * @param rank The new rank.
     */
    void set_rank(std::int32_t rank);

    /**
     * @brief Logs a message using the specified verbosity level.
     *
     * Formats and outputs a message if the verbosity level is high enough.
     *
     * @tparam Args Types of the message components, must support the `<<` operator.
     * @param level The verbosity level of the message.
     * @param args The components of the message to log.
     */
    template <typename... Args>
    void log(LOG_LEVEL level, Args const&... args) {
        if (static_cast<std::uint32_t>(level_) < static_cast<std::uint32_t>(level)) {
            return;
        }
        std::ostringstream ss;
        (ss << ... << args);
        do_log(level, std::move(ss));
    }

    /**
     * @brief Logs a print message.
     *
     * @tparam Args Types of the message components.
     * @param args The components of the message to log.
     */
    template <typename... Args>
    void print(Args const&... args) {
        log(LOG_LEVEL::PRINT, std::forward<Args const&>(args)...);
    }

    /**
     * @brief Logs a warning message.
     *
     * @tparam Args Types of the message components.
     * @param args The components of the message to log.
     */
    template <typename... Args>
    void warn(Args const&... args) {
        log(LOG_LEVEL::WARN, std::forward<Args const&>(args)...);
    }

    /**
     * @brief Logs an informational message.
     *
     * @tparam Args Types of the message components.
     * @param args The components of the message to log.
     */
    template <typename... Args>
    void info(Args const&... args) {
        log(LOG_LEVEL::INFO, std::forward<Args const&>(args)...);
    }

    /**
     * @brief Logs a debug message.
     *
     * @tparam Args Types of the message components.
     * @param args The components of the message to log.
     */
    template <typename... Args>
    void debug(Args const&... args) {
        log(LOG_LEVEL::DEBUG, std::forward<Args const&>(args)...);
    }

    /**
     * @brief Logs a trace message.
     *
     * @tparam Args Types of the message components.
     * @param args The components of the message to log.
     */
    template <typename... Args>
    void trace(Args const&... args) {
        log(LOG_LEVEL::TRACE, std::forward<Args const&>(args)...);
    }

  protected:
    /**
     * @brief Constructs a logger with a known rank.
     * @param rank The rank of the calling process.
     * @param options Configuration options.
     */
    Logger(std::int32_t rank, config::Options options);

    /**
     * @brief Constructs a logger without a known rank (defaults to `-1`).
     * @param options Configuration options.
     */
    explicit Logger(config::Options options);

    /**
     * @brief Returns a unique thread ID for the current thread.
     *
     * @return A unique ID for the current thread.
     */
    virtual std::uint32_t get_thread_id() {
        auto const tid = std::this_thread::get_id();

        // To avoid large IDs, we map the thread ID to an unique counter.
        auto const [name, inserted] =
            thread_id_names.insert({tid, thread_id_names_counter});
        if (inserted) {
            ++thread_id_names_counter;
        }
        return name->second;
    }

    /**
     * @brief Handles the logging of a messages.
     *
     * This base implementation prepend the rank and thread id to the message
     * and print it to `std::cout`.
     *
     * Override this method in a derived classes to customize logging behavior.
     *
     * @param level The verbosity level of the message.
     * @param ss The formatted message as a string stream.
     */
    virtual void do_log(LOG_LEVEL level, std::ostringstream&& ss) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream full_log_msg;
        full_log_msg << "[" << level_name(level) << ":" << rank_ << ":" << get_thread_id()
                     << ":" << Clock::now() << "] " << ss.str();
        std::cout << full_log_msg.str() << std::endl;
    }

  private:
    std::mutex mutex_;
    std::int32_t rank_;  ///< Guarded by `mutex_`.
    LOG_LEVEL const level_;

    /// Counter used by `std::this_thread::get_id()` to abbreviate the large
    /// number returned by `std::this_thread::get_id()`.
    std::uint32_t thread_id_names_counter{0};

    /// Thread record mapping thread IDs to their shorten names.
    std::unordered_map<std::thread::id, std::uint32_t> thread_id_names;
};

}  // namespace rapidsmpf
