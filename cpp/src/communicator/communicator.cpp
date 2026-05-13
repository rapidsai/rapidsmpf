/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/options.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {
namespace {
Communicator::Logger::LOG_LEVEL level_from_string(std::string const& str) {
    auto const value = str.empty() ? std::string{communicator::LogOption.default_val}
                                   : to_upper(trim(str));
    for (std::uint32_t i = 0; i < Communicator::Logger::LOG_LEVEL_NAMES.size(); ++i) {
        auto level = static_cast<Communicator::Logger::LOG_LEVEL>(i);
        if (value == Communicator::Logger::level_name(level)) {
            return level;
        }
    }
    std::stringstream ss;
    ss << "RAPIDSMPF_LOG - unknown value: \"" << value << "\", valid choices: { ";
    for (auto const& name : Communicator::Logger::LOG_LEVEL_NAMES) {
        ss << name << " ";
    }
    ss << "}";
    throw std::invalid_argument(ss.str());
}
}  // namespace

Communicator::Logger::Logger(Rank rank, config::Options options)
    : rank_{rank},
      level_(options.get<LOG_LEVEL>(communicator::LogOption.key, level_from_string)) {};


}  // namespace rapidsmpf
