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

#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/option.hpp>

/**
 * @namespace rapidsmp
 * @brief RAPIDS Multi-Processor interfaces.
 */
namespace rapidsmp {

/**
 * @typedef Rank
 * @brief The rank of a node (e.g. the rank of a MPI process).
 *
 * @note Ranks are always consecutive integers from zero to the total number of ranks.
 */
using Rank = int;

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
     * Encapsulates the concept of an asynchronous operation, allowing users to query or
     * wait for completion.
     */
    class Future {
      public:
        Future() = default;
        virtual ~Future() noexcept = default;
        Future(Future&&) = default;  ///< Movable.
        Future(Future&) = delete;  ///< Not copyable.
    };

    /**
     * @brief Logger base class.
     *
     * To control the verbosity level, set the environment variable `RAPIDSMP_LOG`:
     *   - `0`: Disable all logging.
     *   - `1`: Enable warnings only (default).
     *   - `2`: Enable warnings and informational messages.
     */
    class Logger {
      public:
        /**
         * @brief Construct a new logger.
         *
         * Initializes the logger with a given communicator and verbosity level.
         * The verbosity level is determined by the environment variable `RAPIDSMP_LOG`,
         * defaulting to `1` if not set.
         *
         * @param comm The @ref Communicator to use.
         */
        Logger(Communicator* comm)  // TODO: support writing to a file.
            : comm_{comm}, level_{getenv_or("RAPIDSMP_LOG", 1)} {};
        virtual ~Logger() noexcept = default;

        /**
         * @brief Logs a warning message.
         *
         * Formats and outputs a warning message if the verbosity level is `1` or higher.
         *
         * @tparam Args Types of the message components, must support the << operator.
         * @param args The components of the message to log.
         */
        template <typename... Args>
        void warn(Args const&... args) {
            if (level_ < 1) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            std::ostringstream ss;
            (ss << ... << args);
            do_warn(std::move(ss));
        }

        /**
         * @brief Logs an informational message.
         *
         * Formats and outputs an informational message if the verbosity level is `2`.
         *
         * @tparam Args Types of the message components, must support the << operator.
         * @param args The components of the message to log.
         */
        template <typename... Args>
        void info(Args const&... args) {
            if (level_ < 2) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex_);
            std::ostringstream ss;
            (ss << ... << args);
            do_info(std::move(ss));
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
         * @brief Handles the logging of warning messages.
         *
         * Outputs a formatted warning message to `std::cout`. This method can be
         * overridden in derived classes to customize logging behavior.
         *
         * @param ss The formatted warning message as a string stream.
         */
        virtual void do_warn(std::ostringstream&& ss) {
            std::cout << "[WARN:" << comm_->rank() << ":" << get_thread_id() << "] "
                      << ss.str() << std::endl;
        }

        /**
         * @brief Handles the logging of informational messages.
         *
         * Outputs a formatted informational message to `std::cout`. This method can be
         * overridden in derived classes to customize logging behavior.
         *
         * @param ss The formatted informational message as a string stream.
         */
        virtual void do_info(std::ostringstream&& ss) {
            std::cout << "[INFO:" << comm_->rank() << ":" << get_thread_id() << "] "
                      << ss.str() << std::endl;
        }

        /**
         * @brief Get a reference to the class mutex.
         *
         * @return Reference to the mutex.
         */
        std::mutex& mutex() {
            return mutex_;
        }

        /**
         * @brief Get the communicator used by the logger.
         *
         * @return Pointer to the Communicator instance.
         */
        Communicator* get_communicator() const {
            return comm_;
        }

        /**
         * @brief Get the verbosity level of the logger.
         *
         * Levels:
         *  - `0`: Disable all logging.
         *  - `1`: Enable warnings only (default).
         *  - `2`: Enable warnings and informational messages.
         *
         * @return The verbosity level.
         */
        int verbosity_level() const {
            return level_;
        }


      private:
        std::mutex mutex_;
        Communicator* comm_;
        int const level_;

        /// Counter used by `std::this_thread::get_id()` to abbreviate the large number
        /// returned by `std::this_thread::get_id()`.
        std::uint32_t thread_id_names_counter{0};

        /// Thread name record mapping thread IDs to their shorten names.
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
    [[nodiscard]] virtual int nranks() const = 0;

    /**
     * @brief Sends a host message to a specific rank.
     *
     * @param msg Unique pointer to the message data (host memory).
     * @param rank The destination rank.
     * @param tag Message tag for identification.
     * @param stream CUDA stream used for device memory operations.
     * @param br Buffer resource used to allocate the received message.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> send(
        std::unique_ptr<std::vector<uint8_t>> msg,
        Rank rank,
        int tag,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) = 0;


    /**
     * @brief Sends a message (device or host) to a specific rank.
     *
     * @param msg Unique pointer to the message data (Buffer).
     * @param rank The destination rank.
     * @param tag Message tag for identification.
     * @param stream CUDA stream used for device memory operations.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream
    ) = 0;

    /**
     * @brief Receives a message from a specific rank.
     *
     * @param rank The source rank.
     * @param tag Message tag for identification.
     * @param recv_buffer The receive buffer.
     * @param stream CUDA stream used for device memory operations.
     * @return A unique pointer to a `Future` representing the asynchronous operation.
     */
    [[nodiscard]] virtual std::unique_ptr<Future> recv(
        Rank rank,
        int tag,
        std::unique_ptr<Buffer> recv_buffer,
        rmm::cuda_stream_view stream
    ) = 0;

    /**
     * @brief Receives a message from any rank (blocking).
     *
     * @param tag Message tag for identification.
     * @return A pair containing the message data (host memory) and the rank of the
     * sender.
     */
    [[nodiscard]] virtual std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(
        int tag
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

/**
 * @brief Overloads the stream insertion operator for the Communicator class.
 *
 * This function allows a description of a Communicator to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, Communicator const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmp
