/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>

/**
 * @namespace rapidsmpf
 * @brief RAPIDS Multi-Processor interfaces.
 */
namespace rapidsmpf {

/**
 * @typedef Rank
 * @brief The rank of a node (e.g. the rank of a MPI process), or world size (total number
 * of ranks).
 *
 * @note Ranks are always consecutive integers from zero to the total number of ranks.
 */
using Rank = std::int32_t;

/**
 * @typedef OpID
 * @brief Operation ID defined by the user. This allows users to concurrently execute
 * multiple operations, and each operation will be identified by its OpID.
 *
 * @note This limits the total number of concurrent operations to 2^8
 */
using OpID = std::uint8_t;

/**
 * @typedef StageID
 * @brief Identifier for a stage of a communication operation.
 */
using StageID = std::uint8_t;

/**
 * @brief A tag used for identifying messages in a communication operation.
 *
 * @note The tag is a 32-bit integer, with the following layout:
 * bits     |31:16| 15:8 | 7:0 |
 * value    |empty|  op  |stage|
 */
class Tag {
  public:
    /**
     * @typedef StorageT
     * @brief The physical data type to store the tag
     */
    using StorageT = std::int32_t;

    /// @brief Number of bits for the stage ID
    static constexpr int stage_id_bits{sizeof(StageID) * 8};

    /// @brief Mask for the stage ID
    static constexpr StorageT stage_id_mask{(1 << stage_id_bits) - 1};

    /// @brief Number of bits for the operation ID
    static constexpr int op_id_bits{sizeof(OpID) * 8};

    /// @brief Mask for the operation ID
    static constexpr StorageT op_id_mask{
        ((1 << (op_id_bits + stage_id_bits)) - 1) ^ stage_id_mask
    };

    /**
     * @brief Constructs a tag
     *
     * @param op The operation ID
     * @param stage The stage ID
     */
    constexpr Tag(OpID const op, StageID const stage)
        : tag_{
            (static_cast<StorageT>(op) << stage_id_bits) | static_cast<StorageT>(stage)
        } {}

    /**
     * @brief Returns the max number of bits used for the tag
     * @return bit length
     */
    [[nodiscard]] static constexpr size_t bit_length() noexcept {
        return op_id_bits + stage_id_bits;
    }

    /**
     * @brief Returns the max value of the tag
     * @return max value
     */
    [[nodiscard]] static constexpr StorageT max_value() noexcept {
        return (1 << bit_length()) - 1;
    }

    /**
     * @brief Returns the int32 view of the tag
     * @return int32 view of the tag
     */
    constexpr operator StorageT() const noexcept {
        return tag_;
    }

    /**
     * @brief Extracts the operation ID from the tag
     * @return The operation ID
     */
    [[nodiscard]] constexpr OpID op() const noexcept {
        return (tag_ & op_id_mask) >> stage_id_bits;
    }

    /**
     * @brief Extracts the stage ID from the tag
     * @return The stage ID
     */
    [[nodiscard]] constexpr StageID stage() const noexcept {
        return tag_ & stage_id_mask;
    }

  private:
    StorageT const tag_;
};

/**
 * @brief Abstract base class for a communication mechanism between nodes.
 *
 * Provides an interface for sending and receiving messages between nodes, supporting
 * asynchronous operations, GPU data transfers, and custom logging. Implementations must
 * define the virtual methods to enable specific communication backends.
 */
class Communicator {
  public:
    /**
     * @brief Abstract base class for asynchronous operation within the communicator.
     *
     * Encapsulates the concept of an asynchronous operation, allowing users to query
     * or wait for completion.
     */
    class Future {
      public:
        Future() = default;
        virtual ~Future() noexcept = default;
        Future(Future&&) = default;  ///< Movable.
        Future(Future&) = delete;  ///< Not copyable.
    };

    /**
     * @brief A logger base class for handling different levels of log messages.
     *
     * The logger class provides various logging methods with different verbosity levels.
     * It ensures thread-safety using a mutex and allows filtering of log messages
     * based on the configured verbosity level.
     *
     * TODO: support writing to a file.
     */
    class Logger {
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
        static constexpr const char* level_name(LOG_LEVEL level) {
            auto index = static_cast<std::size_t>(level);
            return index < LOG_LEVEL_NAMES.size() ? LOG_LEVEL_NAMES[index] : "UNKNOWN";
        }

        /**
         * @brief Construct a new logger.
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
         * @param comm The `Communicator` to use.
         * @param options Configuration options.
         */
        Logger(Communicator* comm, config::Options options);
        virtual ~Logger() noexcept = default;

        /**
         * @brief Get the verbosity level of the logger.
         *
         * @return The verbosity level.
         */
        LOG_LEVEL verbosity_level() const {
            return level_;
        }

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
            std::ostringstream full_log_msg;
            full_log_msg << "[" << level_name(level) << ":" << comm_->rank() << ":"
                         << get_thread_id() << "] " << ss.str();
            std::lock_guard<std::mutex> lock(mutex_);
            std::cout << full_log_msg.str() << std::endl;
        }

        /**
         * @brief Get the communicator used by the logger.
         *
         * @return Pointer to the Communicator instance.
         */
        Communicator* get_communicator() const {
            return comm_;
        }

      private:
        std::mutex mutex_;
        Communicator* comm_;
        LOG_LEVEL const level_;

        /// Counter used by `std::this_thread::get_id()` to abbreviate the large
        /// number returned by `std::this_thread::get_id()`.
        std::uint32_t thread_id_names_counter{0};

        /// Thread record mapping thread IDs to their shorten names.
        std::unordered_map<std::thread::id, std::uint32_t> thread_id_names;
    };

  protected:
    Communicator() = default;

  public:
    virtual ~Communicator() noexcept = default;

    /**
     * @brief Retrieves the rank of the current node.
     * @return The rank of the node.
     */
    [[nodiscard]] virtual Rank rank() const = 0;

    /**
     * @brief Retrieves the total number of ranks.
     * @return The total number of ranks.
     */
    [[nodiscard]] virtual Rank nranks() const = 0;

    /**
     * @brief Sends a host message to a specific rank.
     *
     * @param msg Unique pointer to the message data (host memory).
     * @param rank The destination rank.
     * @param tag Message tag for identification.
     * @param br Buffer resource used to allocate the received message.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> send(
        std::unique_ptr<std::vector<uint8_t>> msg, Rank rank, Tag tag, BufferResource* br
    ) = 0;


    /**
     * @brief Sends a message (device or host) to a specific rank.
     *
     * @param msg Unique pointer to the message data (Buffer).
     * @param rank The destination rank.
     * @param tag Message tag for identification.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     *
     * @warning The caller is responsible to ensure the underlying `Buffer` allocation
     * and data are already valid before calling, for example, when a CUDA allocation
     * and/or copy are done asynchronously. Specifically, the caller should ensure
     * `Buffer::is_ready()` returns true before calling this function, if not, a
     * warning is printed and the application will terminate.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, Tag tag
    ) = 0;

    /**
     * @brief Receives a message from a specific rank.
     *
     * @param rank The source rank.
     * @param tag Message tag for identification.
     * @param recv_buffer The receive buffer.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     *
     * @warning The caller is responsible to ensure the underlying `Buffer` allocation
     * is already valid before calling, for example, when a CUDA allocation
     * and/or copy are done asynchronously. Specifically, the caller should ensure
     * `Buffer::is_ready()` returns true before calling this function, if not, a
     * warning is printed and the application will terminate.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> recv(
        Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
    ) = 0;

    /**
     * @brief Receives a message from any rank (blocking).
     *
     * @param tag Message tag for identification.
     * @return A pair containing the message data (host memory) and the rank of the
     * sender.
     * @note If no message is available this is indicated by returning
     * a `nullptr` in the first slot of the pair.
     */
    [[nodiscard]] virtual std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(
        Tag tag
    ) = 0;

    /**
     * @brief Tests for completion of multiple futures.
     *
     * @param future_vector Vector of Future objects.
     * @return Indices of completed futures.
     */
    std::vector<std::size_t> virtual test_some(
        std::vector<std::unique_ptr<Future>> const& future_vector
    ) = 0;

    /**
     * @brief Tests for completion of multiple futures in a map.
     *
     * @param future_map Map of futures identified by keys.
     * @return Keys of completed futures.
     */
    std::vector<std::size_t> virtual test_some(
        std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
            future_map
    ) = 0;

    /**
     * @brief Retrieves GPU data associated with a completed future.
     *
     * @param future The completed future.
     * @return A unique pointer to the GPU data buffer.
     */
    [[nodiscard]] std::unique_ptr<Buffer> virtual get_gpu_data(
        std::unique_ptr<Communicator::Future> future
    ) = 0;

    /**
     * @brief Retrieves the logger associated with this communicator.
     * @return Reference to the logger.
     */
    [[nodiscard]] virtual Logger& logger() = 0;

    /**
     * @brief Provides a string representation of the communicator.
     * @return A string describing the communicator.
     */
    [[nodiscard]] virtual std::string str() const = 0;
};

/// @brief Whether RapidsMPF was built with the UCXX Communicator.
#ifdef RAPIDSMPF_HAVE_UCXX
constexpr bool COMM_HAVE_UCXX = true;
#else
constexpr bool COMM_HAVE_UCXX = false;
#endif

/// @brief Whether RapidsMPF was built with the MPI Communicator.
#ifdef RAPIDSMPF_HAVE_MPI
constexpr bool COMM_HAVE_MPI = true;
#else
constexpr bool COMM_HAVE_MPI = false;
#endif

/**
 * @brief Overloads the stream insertion operator for the Communicator class.
 *
 * This function allows a description of a Communicator to be written to an output
 * stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, Communicator const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf
