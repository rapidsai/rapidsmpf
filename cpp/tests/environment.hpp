/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/progress_thread.hpp>

class Environment : public ::testing::Environment {
  public:
    Environment(int argc, char** argv);

    void SetUp() override;

    void TearDown() override;

    void barrier();

    std::shared_ptr<rapidsmp::Communicator> comm_;
    std::shared_ptr<rapidsmp::ProgressThread> progress_thread_;

  private:
    int argc_;
    char** argv_;
    MPI_Comm mpi_comm_;
};

extern Environment* GlobalEnvironment;
