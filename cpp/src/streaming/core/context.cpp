/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::streaming {


Context::Context(
    config::Options options,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    std::unique_ptr<coro::thread_pool> executor,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
)
    : options_{std::move(options)},
      comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      executor_{std::move(executor)},
      br_{std::move(br)},
      statistics_{std::move(statistics)} {
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "comm cannot be NULL");
    RAPIDSMPF_EXPECTS(progress_thread_ != nullptr, "progress_thread cannot be NULL");
    RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "statistics cannot be NULL");
}

Context::Context(
    config::Options options,
    std::shared_ptr<Communicator> comm,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
)
    : Context(
          options,
          comm,
          std::make_shared<ProgressThread>(comm->logger(), statistics),
          coro::thread_pool::make_unique(
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
          br,
          statistics
      ) {}

config::Options Context::get_options() const noexcept {
    return options_;
}

std::shared_ptr<Communicator> Context::comm() const noexcept {
    return comm_;
}

Communicator::Logger& Context::logger() const noexcept {
    return comm_->logger();
}

std::shared_ptr<ProgressThread> Context::progress_thread() const noexcept {
    return progress_thread_;
}

std::unique_ptr<coro::thread_pool>& Context::executor() noexcept {
    return executor_;
}

BufferResource* Context::br() const noexcept {
    return br_;
}

std::shared_ptr<Statistics> Context::statistics() const noexcept {
    return statistics_;
}

std::shared_ptr<Channel> Context::create_channel() const noexcept {
    return std::shared_ptr<Channel>(new Channel());
}
}  // namespace rapidsmpf::streaming
