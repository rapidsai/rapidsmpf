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

}  // namespace rapidsmpf
