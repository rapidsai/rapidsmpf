/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/coro_executor.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::streaming {

CoroThreadPoolExecutor::CoroThreadPoolExecutor(
    std::uint32_t num_streaming_threads, std::shared_ptr<Statistics> statistics
)
    : executor_{coro::thread_pool::make_unique(coro::thread_pool::options{
          .thread_count            = num_streaming_threads,
          .on_thread_start_functor = [](std::size_t idx) {
#if defined(__linux__) || defined(__APPLE__)
              // Linux comm limit is 15 chars + NUL. 'rmpf-coro-<idx>' fits
              // for idx up to 4 digits, which comfortably exceeds any
              // realistic thread count.
              char name[16] = {};
              std::snprintf(name, sizeof(name), "rmpf-coro-%zu", idx);
#if defined(__linux__)
              pthread_setname_np(pthread_self(), name);
#else  // __APPLE__: names only the calling thread, no pthread_t argument.
              pthread_setname_np(name);
#endif
#else
              (void)idx;
#endif
          },
      })},
      statistics_{std::move(statistics)},
      creator_thread_id_{std::this_thread::get_id()} {
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
              [](std::string const& s) -> std::uint32_t {
                  auto const v = parse_string<int>(s);
                  if (v > 0) {
                      return static_cast<std::uint32_t>(v);
                  }
                  RAPIDSMPF_FAIL(
                      "num_streaming_threads must be a positive integer",
                      std::invalid_argument
                  );
              }
          ),
          std::move(statistics)
      ) {}

CoroThreadPoolExecutor::~CoroThreadPoolExecutor() noexcept {
    shutdown();
}

void CoroThreadPoolExecutor::shutdown() noexcept {
    // Only allow shutdown to occur once.
    if (!is_shutdown_.exchange(true, std::memory_order::acq_rel)) {
        auto const tid = std::this_thread::get_id();
        RAPIDSMPF_EXPECTS_FATAL(
            tid == creator_thread_id_,
            "CoroThreadPoolExecutor::shutdown() called from a different "
            "thread than the one that constructed the executor"
        );
        executor_->shutdown();
    }
}

std::unique_ptr<coro::thread_pool>& CoroThreadPoolExecutor::get() noexcept {
    return executor_;
}
}  // namespace rapidsmpf::streaming
