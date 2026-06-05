/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/runtime.hpp>
#include <rapidsmpf/streaming/core/coro_executor.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace {

std::uint32_t get_num_streaming_threads(rapidsmpf::Runtime* runtime) {
    RAPIDSMPF_EXPECTS(runtime != nullptr, "runtime cannot be NULL");
    return runtime->options().get<std::uint32_t>(
        "num_streaming_threads", [](std::string const& s) -> std::uint32_t {
            auto const v = rapidsmpf::parse_string<int>(s);
            if (v > 0) {
                return static_cast<std::uint32_t>(v);
            }
            RAPIDSMPF_FAIL(
                "num_streaming_threads must be a positive integer", std::invalid_argument
            );
        }
    );
}

}  // namespace

namespace rapidsmpf::streaming {

CoroThreadPoolExecutor::CoroThreadPoolExecutor(
    std::uint32_t num_streaming_threads, std::shared_ptr<Runtime> runtime
)
    : executor_{coro::thread_pool::make_unique(
          coro::thread_pool::options{.thread_count = num_streaming_threads}
      )},
      runtime_{std::move(runtime)},
      creator_thread_id_{std::this_thread::get_id()} {
    RAPIDSMPF_EXPECTS(runtime_ != nullptr, "runtime cannot be NULL");
    RAPIDSMPF_EXPECTS(
        num_streaming_threads > 0,
        "num_streaming_threads must be a positive integer",
        std::invalid_argument
    );
}

CoroThreadPoolExecutor::CoroThreadPoolExecutor(std::shared_ptr<Runtime> runtime)
    : executor_{coro::thread_pool::make_unique(
          coro::thread_pool::options{
              .thread_count = get_num_streaming_threads(runtime.get())
          }
      )},
      runtime_{std::move(runtime)},
      creator_thread_id_{std::this_thread::get_id()} {}

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
