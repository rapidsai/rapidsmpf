/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {
namespace {
Logger::LOG_LEVEL level_from_string(std::string const& str) {
    auto const trimmed = to_upper(trim(str));
    for (std::uint32_t i = 0; i < Logger::LOG_LEVEL_NAMES.size(); ++i) {
        auto level = static_cast<Logger::LOG_LEVEL>(i);
        if (trimmed == Logger::level_name(level)) {
            return level;
        }
    }
    std::stringstream ss;
    ss << "RAPIDSMPF_LOG - unknown value: \"" << trimmed << "\", valid choices: { ";
    for (auto const& name : Logger::LOG_LEVEL_NAMES) {
        ss << name << " ";
    }
    ss << "}";
    throw std::invalid_argument(ss.str());
}
}  // namespace

Logger::Logger(LOG_LEVEL level, std::string name)
    : level_(level), name_(std::move(name)) {}

std::shared_ptr<Logger> Logger::create(LOG_LEVEL level, std::string name) {
    return std::shared_ptr<Logger>(new Logger(level, std::move(name)));
}

std::shared_ptr<Logger> Logger::from_options(config::Options options) {
    auto const level = options.get<LOG_LEVEL>("log", level_from_string);
    return std::shared_ptr<Logger>(new Logger(level, "unknown"));
}

void Logger::set_name(std::string name) {
    std::lock_guard<std::mutex> lock(mutex_);
    name_ = std::move(name);
}

communicator_aborted::communicator_aborted(Rank origin, std::string const& reason)
    : std::
          runtime_error{"communicator aborted by rank " + std::to_string(origin) + ": " + reason},
      origin_{origin},
      reason_{reason} {}

Rank communicator_aborted::origin() const noexcept {
    return origin_;
}

std::string const& communicator_aborted::reason() const noexcept {
    return reason_;
}

namespace detail {

/**
 * @brief The abort state of a communicator.
 *
 * Set by the first abort only, and never cleared. All methods are thread-safe.
 */
class AbortState {
  public:
    /**
     * @brief Record an abort, unless one has already been recorded.
     *
     * @param origin The rank that aborted the communicator, or empty for this rank.
     * @param reason The reason given by `origin`.
     * @return True if this abort was recorded, false if one was already recorded.
     */
    bool set(std::optional<Rank> origin, std::string reason) noexcept {
        bool expected = false;
        if (!claimed_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel, std::memory_order_relaxed
            ))
        {
            return false;
        }
        origin_ = origin;
        reason_ = std::move(reason);
        is_set_.store(true, std::memory_order_release);
        return true;
    }

    /// @return Whether an abort has been recorded.
    [[nodiscard]] bool is_set() const noexcept {
        return is_set_.load(std::memory_order_acquire);
    }

    /// @return The rank that aborted, or empty for this rank. Valid once `is_set()`.
    [[nodiscard]] std::optional<Rank> origin() const noexcept {
        return origin_;
    }

    /// @return The reason given by the rank that aborted. Valid once `is_set()`.
    [[nodiscard]] std::string const& reason() const noexcept {
        return reason_;
    }

  private:
    std::atomic<bool> claimed_{false};  ///< Whether a writer claimed the state.
    std::atomic<bool> is_set_{false};  ///< Whether the state has been published.
    std::optional<Rank> origin_;
    std::string reason_;
};

}  // namespace detail

namespace {
/// @brief How long `FutureGraveyard::release()` waits for the graves, in total.
constexpr auto graveyard_release_timeout = std::chrono::seconds{1};
}  // namespace

FutureGraveyard::FutureGraveyard(std::shared_ptr<detail::AbortState> abort_state)
    : abort_state_{std::move(abort_state)} {}

FutureGraveyard::~FutureGraveyard() noexcept {
    // Not released, so whether they completed is unknown: never free a buffer that may
    // still be accessed.
    for (auto& grave : graves_) {
        std::ignore = grave.release();
    }
}

void FutureGraveyard::bury(std::unique_ptr<Grave> grave) noexcept {
    try {
        abort_state_->set(
            std::nullopt,
            "a future was destroyed while its communication was in flight, e.g., "
            "because its operation failed"
        );
    } catch (...) {
        abort_state_->set(std::nullopt, std::string{});
    }
    std::lock_guard lock(mutex_);
    if (grave == nullptr) {
        return;
    }
    if (released_) {
        std::ignore = grave.release();
        return;
    }
    try {
        graves_.push_back(std::move(grave));
    } catch (...) {
        // Never free a buffer that may still be accessed, leak instead.
        std::ignore = grave.release();
    }
}

bool FutureGraveyard::empty() const noexcept {
    std::lock_guard lock(mutex_);
    return graves_.empty();
}

void FutureGraveyard::release(
    std::function<void()> const& cancel, std::function<void()> const& progress
) noexcept {
    // Take the graves out, so that the communication library is never called while
    // holding the lock, which a future destroyed in one of its callbacks may need.
    std::vector<std::unique_ptr<Grave>> graves;
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        std::swap(graves, graves_);
    }
    if (graves.empty()) {
        return;
    }
    if (cancel) {
        try {
            cancel();
        } catch (...) {
            // Best effort, what does not complete is leaked.
        }
    }
    for (auto& grave : graves) {
        grave->cancel();
    }
    auto const completed = [](auto const& grave) { return grave->completed(); };
    std::erase_if(graves, completed);
    // A single deadline, so that this takes at most about the timeout, however much
    // communication is stuck.
    auto const deadline = std::chrono::steady_clock::now() + graveyard_release_timeout;
    while (!graves.empty() && std::chrono::steady_clock::now() < deadline) {
        if (progress) {
            try {
                progress();
            } catch (...) {
                // Best effort, what does not complete is leaked.
            }
        }
        std::this_thread::yield();
        std::erase_if(graves, completed);
    }
    // Never free a buffer that may still be accessed, leak instead.
    for (auto& grave : graves) {
        std::ignore = grave.release();
    }
}

Communicator::Communicator()
    : abort_state_{std::make_shared<detail::AbortState>()},
      // The constructor of the graveyard is private.
      graveyard_{std::shared_ptr<FutureGraveyard>(new FutureGraveyard(abort_state_))} {}

bool Communicator::is_aborted() const noexcept {
    return abort_state_->is_set();
}

std::shared_ptr<FutureGraveyard> const& Communicator::graveyard() const noexcept {
    return graveyard_;
}

void Communicator::throw_if_aborted() const {
    if (abort_state_->is_set()) {
        throw communicator_aborted(
            abort_state_->origin().value_or(rank()), abort_state_->reason()
        );
    }
}

std::vector<Rank> all_remote_ranks(Communicator const& comm) {
    auto ranks = std::views::iota(Rank{0}, comm.nranks())
                 | std::views::filter([rank = comm.rank()](Rank r) { return r != rank; });
    return {ranks.begin(), ranks.end()};
}

}  // namespace rapidsmpf
