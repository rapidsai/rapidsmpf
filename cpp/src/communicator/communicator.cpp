/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/communicator.hpp>

namespace rapidsmpf {

namespace detail {


Communicator::Logger::LOG_LEVEL level_from_string(std::string const& str) {
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

class LogLevelOption : public config::Option {
  public:
    LogLevelOption() : value{Communicator::Logger::LOG_LEVEL::WARN} {}

    LogLevelOption(std::string const& option_as_string)
        : value{level_from_string(option_as_string)} {}

    ~LogLevelOption() override = default;

    Communicator::Logger::LOG_LEVEL value;
};


}  // namespace detail

Communicator::Logger::Logger(Communicator* comm, config::Options options)
    : comm_{comm}, level_{options.get<detail::LogLevelOption>("log_level")->value} {};


}  // namespace rapidsmpf
