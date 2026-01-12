/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <utility>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::streaming {

namespace {

/**
 * @brief Spill messages until the target amount of memory has been released.
 *
 * Messages are spilled in the order they were inserted into `spillable_messages`.
 * The function stops once the cumulative amount of spilled memory reaches or exceeds
 * `amount`, or when no further spillable messages remain.
 *
 * @param spillable_messages Container holding messages eligible for spilling.
 * @param br Buffer resource used for spill allocations.
 * @param amount Target number of bytes to spill.
 * @return The total number of bytes actually spilled.
 *
 * @todo Support additional spilling strategies (e.g., size-based or priority-based).
 */
std::size_t spill_messages(
    SpillableMessages const& spillable_messages,
    std::shared_ptr<BufferResource> br,
    std::size_t amount
) {
    // Recall that std::map is sorted by key by default, so iteration follows the
    // order in which messages were inserted into `spillable_messages`.
    std::map<SpillableMessages::MessageId, ContentDescription> cds =
        spillable_messages.get_content_descriptions();

    // Iterate over each message and attempt to spill until target amount is reached
    std::size_t total_spilled = 0;
    for (auto const& [id, cd] : cds) {
        if (total_spilled >= amount) {
            break;
        }
        if (cd.spillable()) {
            total_spilled += spillable_messages.spill(id, br.get());
        }
    }
    return total_spilled;
}
}  // namespace

Context::Context(
    config::Options options,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    std::shared_ptr<CoroThreadPoolExecutor> executor,
    std::shared_ptr<BufferResource> br,
    std::shared_ptr<Statistics> statistics
)
    : options_{std::move(options)},
      comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      executor_{std::move(executor)},
      br_{std::move(br)},
      statistics_{std::move(statistics)},
      spillable_messages_{std::make_shared<SpillableMessages>()} {
    RAPIDSMPF_EXPECTS(comm_ != nullptr, "comm cannot be NULL");
    RAPIDSMPF_EXPECTS(progress_thread_ != nullptr, "progress_thread cannot be NULL");
    RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "statistics cannot be NULL");

    // Setup a spilling function.
    spill_function_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t {
            return spill_messages(*spillable_messages_, br_, amount);
        },
        -1  // set priority lower than in the Shuffler and AllGather.
    );
}

Context::Context(
    config::Options options,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<BufferResource> br,
    std::shared_ptr<Statistics> statistics
)
    : Context(
          options,
          comm,
          std::make_shared<ProgressThread>(comm->logger(), statistics),
          std::make_shared<CoroThreadPoolExecutor>(options),
          br,
          statistics
      ) {}

Context::~Context() noexcept {
    br_->spill_manager().remove_spill_function(spill_function_id_);
}

config::Options Context::options() const noexcept {
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

std::shared_ptr<CoroThreadPoolExecutor> Context::executor() const noexcept {
    return executor_;
}

std::shared_ptr<BufferResource> Context::br() const noexcept {
    return br_;
}

std::shared_ptr<Statistics> Context::statistics() const noexcept {
    return statistics_;
}

std::shared_ptr<Channel> Context::create_channel() const noexcept {
    return std::shared_ptr<Channel>(new Channel(spillable_messages()));
}

std::shared_ptr<BoundedQueue> Context::create_bounded_queue(
    std::size_t buffer_size
) const noexcept {
    return std::shared_ptr<BoundedQueue>(new BoundedQueue(buffer_size));
}

std::shared_ptr<SpillableMessages> Context::spillable_messages() const noexcept {
    return spillable_messages_;
}
}  // namespace rapidsmpf::streaming
