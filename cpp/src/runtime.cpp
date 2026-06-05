/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/runtime.hpp>

namespace rapidsmpf {

Runtime::Runtime(
    config::Options options,
    std::shared_ptr<Statistics> statistics,
    std::shared_ptr<Logger> logger
)
    : options_{std::move(options)},
      statistics_{std::move(statistics)},
      logger_{std::move(logger)} {
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "Runtime: statistics cannot be null");
    RAPIDSMPF_EXPECTS(logger_ != nullptr, "Runtime: logger cannot be null");
}

std::shared_ptr<Runtime> Runtime::from_options(config::Options options) {
    auto statistics = Statistics::from_options(options);
    auto logger = Logger::from_options(options);
    return std::shared_ptr<Runtime>(
        new Runtime(std::move(options), std::move(statistics), std::move(logger))
    );
}

void Runtime::reset(config::Options new_options) noexcept {
    options_ = std::move(new_options);
    statistics_ = Statistics::from_options(options_);
    logger_ = Logger::from_options(options_);
}

Statistics& Runtime::statistics() const noexcept {
    return *statistics_;
}

Logger& Runtime::logger() const noexcept {
    return *logger_;
}

}  // namespace rapidsmpf
