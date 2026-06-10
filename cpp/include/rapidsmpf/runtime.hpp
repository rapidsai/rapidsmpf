/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <memory>

#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

/**
 * @brief Central runtime context owning the three shared runtime objects.
 *
 * `Runtime` is the single source of truth for `config::Options`,
 * `Statistics`, and `Logger`.  It must be created first, before any other
 * rapidsmpf object, and must outlive all objects that hold a
 * `std::shared_ptr<Runtime>`.
 *
 * Instances are always managed through `std::shared_ptr` (see `create()`).
 * Any class that needs access to options, statistics, or logging should store
 * a `std::shared_ptr<Runtime>` member and delegate to it.
 *
 * @code{.cpp}
 * auto rt = rapidsmpf::Runtime::from_options(options);
 * rt->logger()->info("Hello from rank ", rt->logger()->verbosity_level());
 * rt->statistics()->add_bytes_stat("foo", 42);
 * @endcode
 */
class Runtime : public std::enable_shared_from_this<Runtime> {
  public:
    /**
     * @brief Create a `Runtime` from configuration options.
     *
     * Constructs a `Statistics` & `Logger` from `options`.
     *
     * @param options Configuration options.
     * @return A `shared_ptr` to the newly constructed Runtime.
     */
    [[nodiscard]] static std::shared_ptr<Runtime> from_options(config::Options options);

    /**
     * @brief Resets runtime with new options.
     *
     * @warning This function is not thread-safe. After resetting, runtime components
     * previously accessed by reference will be invalidated.
     *
     * @param new_options The new options.
     */
    void reset(config::Options new_options) noexcept;

    ~Runtime() noexcept = default;

    Runtime(Runtime const&) = delete;
    Runtime& operator=(Runtime const&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    /**
     * @brief Returns the configuration options.
     *
     * @return A reference to the stored `config::Options`.
     */
    [[nodiscard]] constexpr config::Options& options() noexcept {
        return options_;
    }

    /**
     * @brief Returns the statistics collector.
     *
     * @return A const reference to the `shared_ptr<Statistics>` owned by this
     * Runtime. Always non-null.
     */
    [[nodiscard]] std::shared_ptr<Statistics> const& statistics() const noexcept;

    /**
     * @brief Returns the logger.
     *
     * @return A const reference to the `shared_ptr<Logger>` owned by this
     * Runtime. Always non-null.
     */
    [[nodiscard]] std::shared_ptr<Logger> const& logger() const noexcept;


  private:
    Runtime(
        config::Options options,
        std::shared_ptr<Statistics> statistics,
        std::shared_ptr<Logger> logger
    );

    config::Options options_;
    std::shared_ptr<Statistics> statistics_;
    std::shared_ptr<Logger> logger_;
};

}  // namespace rapidsmpf
