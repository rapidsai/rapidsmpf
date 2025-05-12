/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/communicator.hpp>

namespace rapidsmpf {

namespace detail {


/**
 * @brief Get the verbosity level from the environment variable `RAPIDSMPF_LOG`.
 *
 * This function reads the `RAPIDSMPF_LOG` environment variable, trims whitespace,
 * converts the value to uppercase, and attempts to match it against known logging
 * level names. If the environment variable is not set, the default value `"WARN"`
 * is used.
 *
 * @return The corresponding logging level of type `LOG_LEVEL`.
 *
 * @throws std::invalid_argument If the environment variable contains an unknown
 * value.
 */
Communicator::Logger::LOG_LEVEL level_from_env() {
    auto env = to_upper(trim(getenv_or<std::string>("RAPIDSMPF_LOG", "WARN")));
    for (std::uint32_t i = 0; i < Communicator::Logger::LOG_LEVEL_NAMES.size(); ++i) {
        auto level = static_cast<Communicator::Logger::LOG_LEVEL>(i);
        if (env == Communicator::Logger::level_name(level)) {
            return level;
        }
    }
    std::stringstream ss;
    ss << "RAPIDSMPF_LOG - unknown value: \"" << env << "\", valid choices: { ";
    for (auto const& name : Communicator::Logger::LOG_LEVEL_NAMES) {
        ss << name << " ";
    }
    ss << "}";
    throw std::invalid_argument(ss.str());
}
}  // namespace detail

Communicator::Logger::Logger(Communicator* comm)
    : comm_{comm}, level_{detail::level_from_env()} {};


}  // namespace rapidsmpf
