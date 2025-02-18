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

#include <ucxx/listener.h>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/communicator/ucxx_utils.hpp>

namespace rapidsmp {

namespace ucxx {

namespace {

/**
 * @brief Broadcast listener address of root to all ranks.
 *
 * Broadcast the listener address of root rank to all ranks, so that they are
 * each able to reach and establish an endpoint to the root.
 *
 * @param listener_address object containing the listener address of the root,
 * which will be read from in rank 0 and stored to in all other ranks.
 */
void broadcast_listener_address(std::string& root_worker_address_str) {
    size_t address_size{root_worker_address_str.size()};

    RAPIDSMP_MPI(
        MPI_Bcast(&address_size, sizeof(address_size), MPI_UINT8_T, 0, MPI_COMM_WORLD)
    );

    root_worker_address_str.resize(address_size);

    RAPIDSMP_MPI(MPI_Bcast(
        root_worker_address_str.data(), address_size, MPI_UINT8_T, 0, MPI_COMM_WORLD
    ));
}

}  // namespace

std::shared_ptr<UCXX> init_using_mpi(MPI_Comm mpi_comm) {
    RAPIDSMP_EXPECTS(::rapidsmp::mpi::is_initialized(), "MPI not initialized");

    int rank, nranks;
    RAPIDSMP_MPI(MPI_Comm_rank(mpi_comm, &rank));
    RAPIDSMP_MPI(MPI_Comm_size(mpi_comm, &nranks));

    auto root_listener_address = ListenerAddress{.rank = 0};
    std::string root_worker_address_str{};
    std::shared_ptr<UCXX> comm;
    if (rank == 0) {
        auto ucxx_initialized_rank = init(nullptr, nranks);
        comm = std::make_shared<UCXX>(std::move(ucxx_initialized_rank));

        root_listener_address = comm->listener_address();
        root_worker_address_str =
            std::get<std::shared_ptr<::ucxx::Address>>(root_listener_address.address)
                ->getString();
    }
    broadcast_listener_address(root_worker_address_str);

    if (rank != 0) {
        auto root_worker_address =
            ::ucxx::createAddressFromString(root_worker_address_str);
        auto ucxx_initialized_rank = init(nullptr, nranks, root_worker_address);
        comm = std::make_shared<UCXX>(std::move(ucxx_initialized_rank));
    }

    return comm;
}

}  // namespace ucxx
}  // namespace rapidsmp