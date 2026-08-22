/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <utility>

#include <gtest/gtest.h>
#include <mpi.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/statistics.hpp>

/// @brief Statistics wrapper that clears statistics counters on construction and
/// destruction
///
/// Use this if you want to track statistics for a single test from the shared
/// communicator stats object.
class ClearedStatistics {
  public:
    explicit ClearedStatistics(std::shared_ptr<rapidsmpf::Statistics> statistics)
        : statistics_{std::move(statistics)}, was_enabled_{statistics_->enabled()} {
        statistics_->clear();
        statistics_->enable();
    }

    ~ClearedStatistics() {
        statistics_->clear();
        if (!was_enabled_) {
            statistics_->disable();
        }
    }

    ClearedStatistics(ClearedStatistics const&) = delete;
    ClearedStatistics& operator=(ClearedStatistics const&) = delete;

    [[nodiscard]] rapidsmpf::Statistics* operator->() const noexcept {
        return statistics_.get();
    }

  private:
    std::shared_ptr<rapidsmpf::Statistics> statistics_;
    bool was_enabled_;
};

enum class TestEnvironmentType : int {
    MPI,
    UCXX,
    SINGLE,
};

class Environment : public ::testing::Environment {
  public:
    Environment(int argc, char** argv);

    void SetUp() override;

    void TearDown() override;

    void barrier();

    [[nodiscard]] TestEnvironmentType type() const;

    constexpr rapidsmpf::config::Options& options() {
        return options_;
    }

    std::shared_ptr<rapidsmpf::Communicator> split_comm();

    std::shared_ptr<rapidsmpf::Communicator> comm_;

  private:
    int argc_;
    char** argv_;
    MPI_Comm mpi_comm_;
    std::shared_ptr<rapidsmpf::Communicator> split_comm_{nullptr};
    rapidsmpf::config::Options options_;
};

extern Environment* GlobalEnvironment;
