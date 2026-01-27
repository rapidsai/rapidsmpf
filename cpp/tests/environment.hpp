/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/progress_thread.hpp>

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
    std::shared_ptr<rapidsmpf::ProgressThread> progress_thread_;

  private:
    int argc_;
    char** argv_;
    MPI_Comm mpi_comm_;
    std::shared_ptr<rapidsmpf::Communicator> split_comm_{nullptr};
    rapidsmpf::config::Options options_;
};

extern Environment* GlobalEnvironment;
