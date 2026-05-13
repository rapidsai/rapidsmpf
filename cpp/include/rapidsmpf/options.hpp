/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rapidsmpf {

/**
 * @brief Compile-time descriptor for a single configuration option.
 *
 * Couples an option's lookup key with its default value so the two cannot
 * drift apart. Instances are `inline constexpr` and live in the module
 * sub-namespaces below (e.g. `rapidsmpf::statistics`,
 * `rapidsmpf::buffer_resource`); consult those for the canonical list of
 * options understood by the `from_options` factories.
 *
 * Each descriptor's variable name is suffixed with `Option` to keep it
 * distinct from same-named runtime entities in its module (for example,
 * `rapidsmpf::ucxx::ProgressModeOption` vs the `enum class ProgressMode`).
 *
 * @tparam T The C++ type used to represent the option's parsed value.
 *           For options whose default is a percentage / nbytes / duration
 *           string, T is `std::string_view` and the default is parsed by the
 *           same factory the call site uses for user-supplied values.
 */
template <typename T>
struct OptionDescriptor {
    char const* key;  ///< Null-terminated lookup key passed to `Options::get`.
    T default_val;  ///< Value used when the option is unset.
};

/// @brief Options for `rapidsmpf::Statistics::from_options`.
namespace statistics {
/// @brief Whether statistics tracking is enabled.
inline constexpr OptionDescriptor<std::string_view> EnabledOption{
    .key = "statistics",
    .default_val = "False",
};
}  // namespace statistics

/// @brief Options for `rapidsmpf::PinnedMemoryResource::from_options`.
namespace pinned_memory {
/// @brief Whether pinned host memory is enabled.
inline constexpr OptionDescriptor<bool> EnabledOption{
    .key = "pinned_memory",
    .default_val = false,
};

/// @brief Initial pinned-pool size, applied as
/// `get_host_memory_per_gpu() * InitialPoolSizeFactorOption`.
inline constexpr OptionDescriptor<std::string_view> InitialPoolSizeFactorOption{
    .key = "pinned_initial_pool_size",
    .default_val = "0%",
};

/// @brief Maximum pinned-pool size, applied as
/// `get_host_memory_per_gpu() * MaxPoolSizeFactorOption`.
inline constexpr OptionDescriptor<std::string_view> MaxPoolSizeFactorOption{
    .key = "pinned_max_pool_size",
    .default_val = "80%",
};
}  // namespace pinned_memory

/// @brief Options for `rapidsmpf::BufferResource::from_options` and helpers.
namespace buffer_resource {
/// @brief Device-memory spill limit (nbytes string or percent of total).
inline constexpr OptionDescriptor<std::string_view> SpillDeviceLimitOption{
    .key = "spill_device_limit",
    .default_val = "80%",
};

/// @brief Periodic spill-check interval (duration string or
/// disabled-sentinel).
inline constexpr OptionDescriptor<std::string_view> PeriodicSpillCheckOption{
    .key = "periodic_spill_check",
    .default_val = "1ms",
};

/// @brief CUDA stream-pool size used by the buffer resource.
inline constexpr OptionDescriptor<std::size_t> NumStreamsOption{
    .key = "num_streams",
    .default_val = 16,
};
}  // namespace buffer_resource

/// @brief Options for the streaming subsystem.
namespace streaming {
/// @brief Number of threads in the streaming coroutine pool.
inline constexpr OptionDescriptor<std::uint32_t> NumStreamingThreadsOption{
    .key = "num_streaming_threads",
    .default_val = 1,
};

/// @brief Per-attempt timeout for streaming memory reservations.
inline constexpr OptionDescriptor<std::string_view> MemoryReserveTimeoutOption{
    .key = "memory_reserve_timeout",
    .default_val = "100 ms",
};
}  // namespace streaming

/// @brief Options consumed by `rapidsmpf::Communicator::Logger`.
namespace communicator {
/// @brief Logger verbosity level (string form, one of
/// `Logger::LOG_LEVEL_NAMES`).
inline constexpr OptionDescriptor<std::string_view> LogOption{
    .key = "log",
    .default_val = "WARN",
};
}  // namespace communicator

/// @brief Options consumed by `rapidsmpf::ucxx::init`.
namespace ucxx {
/// @brief UCXX worker progress mode; one of `"blocking"`, `"polling"`,
/// `"thread-blocking"`, `"thread-polling"`.
inline constexpr OptionDescriptor<std::string_view> ProgressModeOption{
    .key = "ucxx_progress_mode",
    .default_val = "thread-blocking",
};
}  // namespace ucxx

}  // namespace rapidsmpf
