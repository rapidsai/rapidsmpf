/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rapidsmpf::defaults {

/**
 * @namespace rapidsmpf::defaults
 * @brief Default values for every configuration option recognised by the
 *        `from_options` factories and option-driven constructors.
 *
 * Each constant is the value that the corresponding `Options::get<T>` lookup
 * falls back to when the option is unset (i.e. the parser receives an empty
 * string). Defaults are stored as primitives or `std::string_view`s so this
 * header has no project-internal dependencies and can be included anywhere
 * cheaply.
 *
 * Enum-typed options (e.g. log level, UCXX progress mode) are kept as their
 * canonical string form and parsed by the same factory that handles
 * user-supplied values, so the default exercises exactly the same code path
 * as user input.
 */

/// @brief Defaults for `rapidsmpf::Statistics::from_options`.
namespace statistics {
/// @brief Default value for the `statistics` option.
inline constexpr std::string_view Enabled = "False";
}  // namespace statistics

/// @brief Defaults for `rapidsmpf::PinnedMemoryResource::from_options`.
namespace pinned_memory {
/// @brief Default value for the `pinned_memory` option.
inline constexpr bool Enabled = false;

/// @brief Default value for the `pinned_initial_pool_size` option.
///
/// Applied as `initial_pool_size = get_host_memory_per_gpu() *
/// InitialPoolSizeFactor`.
inline constexpr std::string_view InitialPoolSizeFactor = "0%";

/// @brief Default value for the `pinned_max_pool_size` option.
///
/// Applied as `max_pool_size = get_host_memory_per_gpu() *
/// MaxPoolSizeFactor`.
inline constexpr std::string_view MaxPoolSizeFactor = "80%";
}  // namespace pinned_memory

/// @brief Defaults for `rapidsmpf::BufferResource::from_options` and its helpers.
namespace buffer_resource {
/// @brief Default value for the `spill_device_limit` option.
///
/// Applied as a fraction of the total device memory.
inline constexpr std::string_view SpillDeviceLimit = "80%";

/// @brief Default value for the `periodic_spill_check` option.
inline constexpr std::string_view PeriodicSpillCheck = "1ms";

/// @brief Default value for the `num_streams` option.
inline constexpr std::size_t NumStreams = 16;
}  // namespace buffer_resource

/// @brief Defaults for the streaming subsystem (used via
/// `rapidsmpf::streaming::Context::from_options`).
namespace streaming {
/// @brief Default value for the `num_streaming_threads` option.
inline constexpr std::uint32_t NumStreamingThreads = 1;

/// @brief Default value for the `memory_reserve_timeout` option.
inline constexpr std::string_view MemoryReserveTimeout = "100 ms";
}  // namespace streaming

/// @brief Defaults consumed by `rapidsmpf::Communicator::Logger`.
namespace communicator {
/// @brief Default value for the `log` option.
///
/// Parsed by the same `level_from_string` helper used for user input;
/// must match one of the names in `Communicator::Logger::LOG_LEVEL_NAMES`.
inline constexpr std::string_view Log = "WARN";
}  // namespace communicator

/// @brief Defaults consumed by `rapidsmpf::ucxx::init`.
namespace ucxx {
/// @brief Default value for the `ucxx_progress_mode` option.
///
/// Parsed by the same lambda used for user input; must match one of
/// `"blocking"`, `"polling"`, `"thread-blocking"`, `"thread-polling"`.
inline constexpr std::string_view ProgressMode = "thread-blocking";
}  // namespace ucxx

}  // namespace rapidsmpf::defaults
