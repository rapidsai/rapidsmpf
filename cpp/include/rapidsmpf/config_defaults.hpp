/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <unordered_map>

namespace rapidsmpf::config {

/**
 * @brief String-form default values for config options.
 *
 * Defaults are stored as strings and parsed through the same factories used
 * for user-supplied values. `Options::get<T>(key, factory)` consults this
 * map automatically: when the user has not supplied a value, the factory
 * receives the registered default string for `key`.
 *
 * Options are not required to have an entry in this map. If no default is
 * registered for a key, the factory receives the empty string.
 *
 * To add a new option, add an entry here and reference it at the call site
 * via `Options::get<T>("<key>", factory)`.
 */
inline const std::unordered_map<std::string, std::string> DEFAULTS{
    {"statistics", "false"},
    {"pinned_memory", "false"},
    {"pinned_initial_pool_size", "0%"},
    {"pinned_max_pool_size", "80%"},
    {"spill_device_limit", "80%"},
    {"periodic_spill_check", "1ms"},
    {"num_streams", "16"},
    {"num_streaming_threads", "1"},
    {"memory_reserve_timeout", "100 ms"},
    {"allow_overbooking_by_default", "true"},
    {"log", "WARN"},
    {"ucxx_progress_mode", "thread-blocking"},
};

}  // namespace rapidsmpf::config
