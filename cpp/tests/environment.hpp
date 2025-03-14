/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/communicator/progress_thread.hpp>

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
