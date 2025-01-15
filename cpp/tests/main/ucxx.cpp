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
#include <rapidsmp/communicator/ucxx.hpp>

#include "../environment.hpp"

Environment* GlobalEnvironment = nullptr;

Environment::Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

static void broadcast_listener_address(rapidsmp::ListenerAddress& listener_address) {
    size_t host_size{listener_address.host.size()};

    RAPIDSMP_MPI(MPI_Bcast(&host_size, sizeof(host_size), MPI_UINT8_T, 0, MPI_COMM_WORLD)
    );

    listener_address.host.resize(host_size);

    RAPIDSMP_MPI(
        MPI_Bcast(listener_address.host.data(), host_size, MPI_UINT8_T, 0, MPI_COMM_WORLD)
    );

    RAPIDSMP_MPI(MPI_Bcast(
        &listener_address.port,
        sizeof(listener_address.port),
        MPI_UINT8_T,
        0,
        MPI_COMM_WORLD
    ));
}

void Environment::SetUp() {
    rapidsmp::mpi::init(&argc_, &argv_);

    RAPIDSMP_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_));
    int rank, nranks;
    RAPIDSMP_MPI(MPI_Comm_rank(mpi_comm_, &rank));
    RAPIDSMP_MPI(MPI_Comm_size(mpi_comm_, &nranks));

    // Ensure CUDA context is created before UCX is initialized.
    cudaFree(0);

    auto root_listener_address = rapidsmp::ListenerAddress{.rank = 0};
    std::shared_ptr<rapidsmp::UCXX> comm;
    if (rank == 0) {
        comm = std::make_shared<rapidsmp::UCXX>(nullptr, nranks);

        root_listener_address = comm->listener_address();
        comm_ = comm;
    }
    broadcast_listener_address(root_listener_address);

    if (rank != 0) {
        comm = std::make_shared<rapidsmp::UCXX>(
            nullptr, nranks, root_listener_address.host, root_listener_address.port
        );
    }

    comm->barrier();

    comm_ = comm;
}

void Environment::TearDown() {
    // Ensure UCXX cleanup before MPI. If this is not done failures related to
    // accessing the CUDA context may be thrown during shutdown.
    comm_ = nullptr;

    RAPIDSMP_MPI(MPI_Comm_free(&mpi_comm_));
    RAPIDSMP_MPI(MPI_Finalize());
}

void Environment::barrier() {
    std::dynamic_pointer_cast<rapidsmp::UCXX>(comm_)->barrier();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GlobalEnvironment = new Environment(argc, argv);
    ::testing::AddGlobalTestEnvironment(GlobalEnvironment);
    return RUN_ALL_TESTS();
}
