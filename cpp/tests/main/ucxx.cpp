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

#include <gtest/gtest.h>
#include <mpi.h>
#include <ucxx/listener.h>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/communicator/ucxx_utils.hpp>

#include "../environment.hpp"

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

void Environment::SetUp() {
    // Ensure CUDA context is created before UCX is initialized.
    cudaFree(nullptr);

    // Explicitly initialize MPI. We can not use rapidsmp::mpi::init as it checks some
    // rapidsmp::MPI communicator specific conditions
    int provided;
    RAPIDSMP_MPI(MPI_Init_thread(&argc_, &argv_, MPI_THREAD_MULTIPLE, &provided));
    RAPIDSMP_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );
    comm_ = rapidsmp::ucxx::init_using_mpi(MPI_COMM_WORLD);
    barrier();
}

void Environment::TearDown() {
    // Ensure UCXX cleanup before MPI. If this is not done failures related to
    // accessing the CUDA context may be thrown during shutdown.
    comm_ = nullptr;

    RAPIDSMP_MPI(MPI_Finalize());
}

void Environment::barrier() {
    std::dynamic_pointer_cast<rapidsmp::ucxx::UCXX>(comm_)->barrier();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}
