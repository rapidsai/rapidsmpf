/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <rapidsmp/communicator/mpi.hpp>

#include "../environment.hpp"

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

void Environment::SetUp() {
    rapidsmp::mpi::init(&argc_, &argv_);

    RAPIDSMP_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_));

    comm_ = std::make_shared<rapidsmp::MPI>(mpi_comm_);
    progress_thread_ = std::make_shared<rapidsmp::ProgressThread>(comm_->logger());
}

void Environment::TearDown() {
    RAPIDSMP_MPI(MPI_Comm_free(&mpi_comm_));
    RAPIDSMP_MPI(MPI_Finalize());
}

void Environment::barrier() {
    RAPIDSMP_MPI(MPI_Barrier(mpi_comm_));
}

std::shared_ptr<rapidsmp::Communicator> Environment::split_comm() {
    // Return cached split communicator if it exists
    if (split_comm_ != nullptr) {
        return split_comm_;
    }

    // Create and cache the new split communicator
    int rank;
    RAPIDSMP_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_Comm mpi_comm;
    RAPIDSMP_MPI(MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &mpi_comm));
    split_comm_ = std::make_shared<rapidsmp::MPI>(mpi_comm);
    return split_comm_;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}
