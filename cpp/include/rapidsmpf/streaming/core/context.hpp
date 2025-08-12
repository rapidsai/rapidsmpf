/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/statistics.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Context for nodes (coroutines) in rapidsmpf.
 */
class Context {
  public:
    /**
     * @brief Full constructor for the Context.
     *
     * All provided pointers must be non-null.
     *
     * @param options Configuration options.
     * @param comm Shared pointer to a communicator.
     * @param progress_thread Shared pointer to a progress thread.
     * @param executor Shared pointer to a coroutine thread pool.
     * @param stream CUDA stream used for asynchronous operations.
     * @param br Shared pointer to a buffer resource.
     * @param statistics Shared pointer to a statistics collector.
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        std::shared_ptr<ProgressThread> progress_thread,
        std::shared_ptr<coro::thread_pool> executor,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics
    )
        : options_{std::move(options)},
          comm_{std::move(comm)},
          progress_thread_{std::move(progress_thread)},
          executor_{std::move(executor)},
          stream_{std::move(stream)},
          br_{std::move(br)},
          statistics_{std::move(statistics)} {
        RAPIDSMPF_EXPECTS(comm_ != nullptr, "comm cannot be NULL");
        RAPIDSMPF_EXPECTS(progress_thread_ != nullptr, "progress_thread cannot be NULL");
        RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
        RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");
        RAPIDSMPF_EXPECTS(statistics_ != nullptr, "statistics cannot be NULL");
    }

    /**
     * @brief Convenience constructor with minimal configuration.
     *
     * Creates a default ProgressThread and coroutine thread pool.
     *
     * @param options Configuration options.
     * @param comm Shared pointer to a communicator.
     * @param stream CUDA stream used for memory operations.
     * @param br Buffer resource used to reserve host memory and perform the move.
     * @param statistics The statistics instance to use (disabled by default).
     */
    Context(
        config::Options options,
        std::shared_ptr<Communicator> comm,
        rmm::cuda_stream_view stream,
        BufferResource* br,
        std::shared_ptr<Statistics> statistics = Statistics::disabled()
    )
        : Context(
              options,
              comm,
              std::make_shared<ProgressThread>(comm->logger(), statistics),
              coro::thread_pool::make_shared(
                  coro::thread_pool::options{
                      .thread_count = options.get<std::uint32_t>(
                          "num_streaming_threads",
                          [](std::string const& s) {
                              if (s.empty()) {
                                  return 1;  // Default number of threads.
                              }
                              if (int v = std::stoi(s); v > 0) {
                                  return v;
                              }
                              throw std::invalid_argument(
                                  "num_streaming_threads must be positive"
                              );
                          }
                      )
                  }
              ),
              stream,
              std::move(br),
              statistics
          ) {}

    /**
     * @brief Returns the configuration options.
     *
     * @return The Options instance.
     */
    config::Options get_options() {
        return options_;
    }

    /**
     * @brief Returns the communicator.
     *
     * @return Shared pointer to the communicator.
     */
    std::shared_ptr<Communicator> comm() {
        return comm_;
    }

    /**
     * @brief Returns the progress thread.
     *
     * @return Shared pointer to the progress thread.
     */
    std::shared_ptr<ProgressThread> progress_thread() {
        return progress_thread_;
    }

    /**
     * @brief Returns the coroutine thread pool.
     *
     * @return Shared pointer to the thread pool.
     */
    std::shared_ptr<coro::thread_pool> executor() {
        return executor_;
    }

    /**
     * @brief Returns the CUDA stream.
     *
     * @return The CUDA stream view.
     */
    rmm::cuda_stream_view stream() {
        return stream_;
    }

    /**
     * @brief Returns the buffer resource.
     *
     * @return Raw pointer to the buffer resource.
     */
    BufferResource* br() {
        return br_;
    }

    /**
     * @brief Returns the statistics collector.
     *
     * @return Shared pointer to the statistics instance.
     */
    std::shared_ptr<Statistics> statistics() {
        return statistics_;
    }

    /**
     * @brief Reserves GPU memory and optionally spills to satisfy the request.
     *
     * @param size Number of bytes to reserve.
     * @return A memory reservation object managing the reserved memory.
     */
    MemoryReservation reserve_and_spill(std::size_t size);

  private:
    config::Options options_;
    std::shared_ptr<Communicator> comm_;
    std::shared_ptr<ProgressThread> progress_thread_;
    std::shared_ptr<coro::thread_pool> executor_;
    rmm::cuda_stream_view stream_;
    BufferResource* br_;
    std::shared_ptr<Statistics> statistics_;
};

}  // namespace rapidsmpf::streaming
