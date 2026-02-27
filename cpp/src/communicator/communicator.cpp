/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {
namespace {
Communicator::Logger::LOG_LEVEL level_from_string(std::string const& str) {
    if (str.empty()) {
        return Communicator::Logger::LOG_LEVEL::WARN;  // Default log level.
    }
    auto trimmed = to_upper(trim(str));
    for (std::uint32_t i = 0; i < Communicator::Logger::LOG_LEVEL_NAMES.size(); ++i) {
        auto level = static_cast<Communicator::Logger::LOG_LEVEL>(i);
        if (trimmed == Communicator::Logger::level_name(level)) {
            return level;
        }
    }
    std::stringstream ss;
    ss << "RAPIDSMPF_LOG - unknown value: \"" << trimmed << "\", valid choices: { ";
    for (auto const& name : Communicator::Logger::LOG_LEVEL_NAMES) {
        ss << name << " ";
    }
    ss << "}";
    throw std::invalid_argument(ss.str());
}
}  // namespace

Communicator::Logger::Logger(Rank rank, config::Options options)
    : rank_{rank}, level_(options.get<LOG_LEVEL>("log", level_from_string)) {};


}  // namespace rapidsmpf
