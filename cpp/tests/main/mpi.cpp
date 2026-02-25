/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <rapidsmpf/communicator/mpi.hpp>

#include "../environment.hpp"

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

TestEnvironmentType Environment::type() const {
    return TestEnvironmentType::MPI;
}

void Environment::SetUp() {
    rapidsmpf::mpi::init(&argc_, &argv_);

    RAPIDSMPF_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_));

    options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());

    comm_ = std::make_shared<rapidsmpf::MPI>(mpi_comm_, options_);
}

void Environment::TearDown() {
    split_comm_ = nullptr;  // Clean up the split communicator.
    comm_ = nullptr;  // Clean up the communicator.

    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm_));
    RAPIDSMPF_MPI(MPI_Finalize());
}

void Environment::barrier() {
    RAPIDSMPF_MPI(MPI_Barrier(mpi_comm_));
}

std::shared_ptr<rapidsmpf::Communicator> Environment::split_comm() {
    // Return cached split communicator if it exists
    if (split_comm_ != nullptr) {
        return split_comm_;
    }

    // Initialize configuration options from environment variables.
    rapidsmpf::config::Options options{rapidsmpf::config::get_environment_variables()};

    // Create and cache the new split communicator
    int rank;
    RAPIDSMPF_MPI(MPI_Comm_rank(mpi_comm_, &rank));
    MPI_Comm split_comm = MPI_COMM_NULL;
    RAPIDSMPF_MPI(MPI_Comm_split(mpi_comm_, rank, 0, &split_comm));
    return std::shared_ptr<rapidsmpf::MPI>(
        new rapidsmpf::MPI(split_comm, options),
        // Don't leak the split handle.
        [comm = split_comm](rapidsmpf::MPI* x) mutable {
            delete x;
            MPI_Comm_free(&comm);
        }
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}
