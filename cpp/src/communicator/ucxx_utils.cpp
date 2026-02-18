/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_device_runtime_api.h>
#include <ucxx/listener.h>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>

namespace rapidsmpf {

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
void broadcast_listener_address(MPI_Comm mpi_comm, std::string& root_worker_address_str) {
    std::size_t address_size{root_worker_address_str.size()};

    RAPIDSMPF_MPI(
        MPI_Bcast(&address_size, sizeof(address_size), MPI_UINT8_T, 0, mpi_comm)
    );

    root_worker_address_str.resize(address_size);

    RAPIDSMPF_MPI(
        MPI_Bcast(root_worker_address_str.data(), address_size, MPI_UINT8_T, 0, mpi_comm)
    );
}

}  // namespace

std::shared_ptr<UCXX> init_using_mpi(
    MPI_Comm mpi_comm, rapidsmpf::config::Options options
) {
    RAPIDSMPF_EXPECTS(::rapidsmpf::mpi::is_initialized(), "MPI not initialized");

    // Ensure CUDA context is created before UCX is initialized.
    cudaFree(nullptr);

    int rank, nranks;
    RAPIDSMPF_MPI(MPI_Comm_rank(mpi_comm, &rank));
    RAPIDSMPF_MPI(MPI_Comm_size(mpi_comm, &nranks));

    auto root_listener_address = ListenerAddress{.rank = 0};
    std::string root_worker_address_str{};
    std::shared_ptr<UCXX> comm;
    if (rank == 0) {
        auto ucxx_initialized_rank = init(nullptr, nranks, std::nullopt, options);
        comm = std::make_shared<UCXX>(std::move(ucxx_initialized_rank), options);

        root_listener_address = comm->listener_address();
        root_worker_address_str =
            std::get<std::shared_ptr<::ucxx::Address>>(root_listener_address.address)
                ->getString();
    }
    broadcast_listener_address(mpi_comm, root_worker_address_str);

    if (rank != 0) {
        auto root_worker_address =
            ::ucxx::createAddressFromString(root_worker_address_str);
        auto ucxx_initialized_rank = init(nullptr, nranks, root_worker_address, options);
        comm = std::make_shared<UCXX>(std::move(ucxx_initialized_rank), options);
    }

    // barrier to complete the bootstrapping process
    comm->barrier();
    return comm;
}

}  // namespace ucxx
}  // namespace rapidsmpf
