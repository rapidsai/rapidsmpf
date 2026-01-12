/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <utility>

#include <rapidsmpf/streaming/core/coro_executor.hpp>

namespace rapidsmpf::streaming {

CoroThreadPoolExecutor::CoroThreadPoolExecutor(
    std::uint32_t num_streaming_threads, std::shared_ptr<Statistics> statistics
)
    : executor_{coro::thread_pool::make_unique(
          coro::thread_pool::options{.thread_count = num_streaming_threads}
      )},
      statistics_{std::move(statistics)} {
    RAPIDSMPF_EXPECTS(
        num_streaming_threads > 0,
        "num_streaming_threads must be a positive integer",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "statistics cannot be NULL");
}

CoroThreadPoolExecutor::CoroThreadPoolExecutor(
    config::Options options, std::shared_ptr<Statistics> statistics
)
    : CoroThreadPoolExecutor(
          options.get<std::uint32_t>(
              "num_streaming_threads",
              [](std::string const& s) {
                  if (s.empty()) {
                      return 1;  // Default number of threads.
                  }
                  if (int v = std::stoi(s); v > 0) {
                      return v;
                  }
                  RAPIDSMPF_FAIL(
                      "num_streaming_threads must be a positive integer",
                      std::invalid_argument
                  );
              }
          ),
          std::move(statistics)
      ) {}

std::unique_ptr<coro::thread_pool>& CoroThreadPoolExecutor::get() noexcept {
    return executor_;
}
}  // namespace rapidsmpf::streaming
