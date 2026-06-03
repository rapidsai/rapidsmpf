/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

Logger::Logger(std::int32_t rank, config::Options options)
    : rank_{rank}, level_(options.get<LOG_LEVEL>("log", level_from_string)) {}

Logger::Logger(config::Options options) : Logger(std::int32_t{-1}, std::move(options)) {}

std::shared_ptr<Logger> Logger::create(std::int32_t rank, config::Options options) {
    return std::shared_ptr<Logger>(new Logger(rank, std::move(options)));
}

std::shared_ptr<Logger> Logger::create(config::Options options) {
    return std::shared_ptr<Logger>(new Logger(std::move(options)));
}

void Logger::set_rank(std::int32_t rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    rank_ = rank;
}

}  // namespace rapidsmpf
