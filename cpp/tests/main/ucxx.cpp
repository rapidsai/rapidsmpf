/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <ucxx/listener.h>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/progress_thread.hpp>

#include "../environment.hpp"

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

TestEnvironmentType Environment::type() const {
    return TestEnvironmentType::UCXX;
}

void Environment::SetUp() {
    // Ensure CUDA context is created before UCX is initialized.
    cudaFree(nullptr);

    // Explicitly initialize MPI. We can not use rapidsmpf::mpi::init as it checks some
    // rapidsmpf::MPI communicator specific conditions
    int provided;
    RAPIDSMPF_MPI(MPI_Init_thread(&argc_, &argv_, MPI_THREAD_MULTIPLE, &provided));
    RAPIDSMPF_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );

    options_ = rapidsmpf::config::Options(rapidsmpf::config::get_environment_variables());
    comm_ = rapidsmpf::ucxx::init_using_mpi(
        MPI_COMM_WORLD, options_, std::make_shared<rapidsmpf::ProgressThread>()
    );
}

void Environment::TearDown() {
    // Ensure UCXX cleanup before MPI. If this is not done failures related to
    // accessing the CUDA context may be thrown during shutdown.
    split_comm_ = nullptr;  // Clean up the split communicator.
    comm_ = nullptr;  // Clean up the communicator.
    RAPIDSMPF_MPI(MPI_Finalize());
}

void Environment::barrier() {
    std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm_)->barrier();
}

std::shared_ptr<rapidsmpf::Communicator> Environment::split_comm() {
    // Return cached split communicator if it exists
    if (split_comm_ != nullptr) {
        return split_comm_;
    }

    // Create and cache the new split communicator
    split_comm_ = std::dynamic_pointer_cast<rapidsmpf::ucxx::UCXX>(comm_)->split();
    return split_comm_;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}
