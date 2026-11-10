/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
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

namespace {
/// @brief How long `release_retired_futures()` waits for the retired futures, in total.
constexpr auto retired_futures_timeout = std::chrono::seconds{1};
}  // namespace

Communicator::~Communicator() noexcept {
    // The backend did not release them, so whether they completed is unknown.
    for (auto& future : retired_futures_) {
        std::ignore = future.release();
    }
}

void Communicator::retire_futures(
    std::vector<std::unique_ptr<Future>>&& futures
) noexcept {
    std::erase(futures, nullptr);
    auto const completed = [this](auto const& future) {
        return retired_future_completed(*future);
    };
    // Nobody can consume the retired receives anymore. Cancel them right away, to narrow
    // the window in which they can match later messages.
    for (auto const& future : futures) {
        cancel_retired_future(*future);
    }
    // Release the futures that completed already, while their stream is surely valid.
    std::erase_if(futures, completed);
    try {
        std::lock_guard lock(retired_futures_mutex_);
        // Release the futures that completed since they were retired.
        std::erase_if(retired_futures_, completed);
        retired_futures_.reserve(retired_futures_.size() + futures.size());
        std::ranges::move(futures, std::back_inserter(retired_futures_));
    } catch (...) {
        // Never free a buffer that may still be accessed, leak instead.
        for (auto& future : futures) {
            std::ignore = future.release();
        }
    }
}

void Communicator::retire_future(std::unique_ptr<Future>&& future) noexcept {
    std::vector<std::unique_ptr<Future>> futures;
    try {
        futures.push_back(std::move(future));
    } catch (...) {
        // Never free a buffer that may still be accessed, leak instead.
        std::ignore = future.release();
        return;
    }
    retire_futures(std::move(futures));
}

void Communicator::release_retired_futures() noexcept {
    std::lock_guard lock(retired_futures_mutex_);
    auto const completed = [this](auto const& future) {
        return retired_future_completed(*future);
    };
    // A single deadline, so that destruction takes at most about the timeout, however
    // many futures are stuck.
    auto const deadline = std::chrono::steady_clock::now() + retired_futures_timeout;
    while (!retired_futures_.empty() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
        std::erase_if(retired_futures_, completed);
    }
    // Never free a buffer that may still be accessed, leak instead.
    for (auto& future : retired_futures_) {
        std::ignore = future.release();
    }
    retired_futures_.clear();
}

std::vector<Rank> all_remote_ranks(Communicator const& comm) {
    auto ranks = std::views::iota(Rank{0}, comm.nranks())
                 | std::views::filter([rank = comm.rank()](Rank r) { return r != rank; });
    return {ranks.begin(), ranks.end()};
}

}  // namespace rapidsmpf
