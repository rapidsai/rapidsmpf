/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <memory>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::streaming {

namespace {

std::size_t get_new_uid() noexcept {
    static std::atomic<std::size_t> uid_counter{0};
    return uid_counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

Context::Context(
    config::Options options,
    std::shared_ptr<Logger> logger,
    std::shared_ptr<CoroThreadPoolExecutor> executor,
    std::shared_ptr<BufferResource> br
)
    : uid_{get_new_uid()},
      creator_thread_id_{std::this_thread::get_id()},
      options_{std::move(options)},
      logger_{std::move(logger)},
      executor_{std::move(executor)},
      br_{std::move(br)},
      spillable_messages_{std::make_shared<SpillableMessages>()} {
    RAPIDSMPF_EXPECTS(logger_ != nullptr, "logger cannot be NULL");
    RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");

    // Setup a spilling function.
    spill_function_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t {
            return spillable_messages_->try_spill(br_.get(), amount);
        },
        -1  // set priority lower than in the Shuffler and AllGather.
    );

    for (auto mem_type : MEMORY_TYPES) {
        memory_[static_cast<std::size_t>(mem_type)] =
            std::make_shared<MemoryReserveOrWait>(options_, mem_type, executor_, br_);
    }
}

Context::Context(
    config::Options options,
    std::shared_ptr<Logger> logger,
    std::shared_ptr<BufferResource> br
)
    : Context(
          options,
          std::move(logger),
          std::make_shared<CoroThreadPoolExecutor>(options),
          br
      ) {}

std::shared_ptr<Context> Context::from_options(
    any_device_resource mr,
    std::shared_ptr<Logger> logger,
    config::Options options,
    std::shared_ptr<Statistics> statistics
) {
    return std::make_shared<Context>(
        options,
        std::move(logger),
        BufferResource::from_options(std::move(mr), options, std::move(statistics))
    );
}

Context::~Context() noexcept {
    shutdown();
}

void Context::shutdown() noexcept {
    // Only allow shutdown to occur once.
    if (!is_shutdown_.exchange(true, std::memory_order::acq_rel)) {
        br_->spill_manager().remove_spill_function(spill_function_id_);
        auto const tid = std::this_thread::get_id();
        RAPIDSMPF_EXPECTS_FATAL(
            tid == creator_thread_id_,
            "Context::shutdown() called from a different thread than the one that "
            "constructed the executor"
        );
        executor_->shutdown();
        spillable_messages_->clear();
    }
}

config::Options Context::options() const noexcept {
    return options_;
}

std::shared_ptr<Logger> const& Context::logger() const noexcept {
    return logger_;
}

std::shared_ptr<CoroThreadPoolExecutor> const& Context::executor() const noexcept {
    return executor_;
}

std::shared_ptr<BufferResource> const& Context::br() const noexcept {
    return br_;
}

std::shared_ptr<MemoryReserveOrWait> const& Context::memory(
    MemoryType mem_type
) const noexcept {
    return memory_[static_cast<std::size_t>(mem_type)];
}

std::shared_ptr<Statistics> Context::statistics() const noexcept {
    return br_->statistics();
}

std::shared_ptr<Channel> Context::create_channel() const noexcept {
    return std::shared_ptr<Channel>(new Channel(spillable_messages()));
}

std::shared_ptr<BoundedQueue> Context::create_bounded_queue(
    std::size_t buffer_size
) const noexcept {
    return std::shared_ptr<BoundedQueue>(new BoundedQueue(buffer_size));
}

std::shared_ptr<SpillableMessages> const& Context::spillable_messages() const noexcept {
    return spillable_messages_;
}

std::size_t Context::uid() const noexcept {
    return uid_;
}

}  // namespace rapidsmpf::streaming
