/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include <cuda_device_runtime_api.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::bootstrap {

std::shared_ptr<ucxx::UCXX> create_ucxx_comm(Backend backend, config::Options options) {
    auto ctx = init(backend);

    // Ensure CUDA context is created before UCX is initialized
    cudaFree(nullptr);

    std::shared_ptr<ucxx::UCXX> comm;

    // Check if root address was provided by parent process (rrun hybrid mode)
    char const* precomputed_address = std::getenv("RAPIDSMPF_ROOT_ADDRESS");

    if (precomputed_address != nullptr) {
        // Parent process already coordinated the root address via PMIx
        // Children skip bootstrap coordination and use the provided address directly
        if (ctx.rank == 0) {
            // Root child creates listener
            auto ucxx_initialized_rank =
                ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
            comm =
                std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
        } else {
            // Worker children connect using provided address
            auto root_worker_address =
                ::ucxx::createAddressFromString(precomputed_address);
            auto ucxx_initialized_rank =
                ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
            comm =
                std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
        }
    } else {
        // Standard bootstrap coordination via put/get/barrier
        if (ctx.rank == 0) {
            // Create root UCXX communicator
            auto ucxx_initialized_rank =
                ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
            comm =
                std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);

            // Get the listener address and publish
            auto listener_address = comm->listener_address();
            auto root_worker_address_str =
                std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address)
                    ->getString();
            put(ctx, "ucxx_root_address", root_worker_address_str);
        }

        // All ranks must barrier to make PMIx put() data visible.
        // For file backend this is a no-op synchronization.
        // For PMIx/Slurm backend this executes PMIx_Fence to exchange data.
        barrier(ctx);

        if (ctx.rank != 0) {
            // Worker ranks retrieve the root address and connect
            auto root_worker_address_str =
                get(ctx, "ucxx_root_address", std::chrono::seconds{30});
            auto root_worker_address =
                ::ucxx::createAddressFromString(root_worker_address_str);

            auto ucxx_initialized_rank =
                ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
            comm =
                std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
        }
    }

    comm->barrier();

    // If root rank and address file path is specified, write the address
    // This is used for parent-mediated coordination in rrun hybrid mode
    if (ctx.rank == 0) {
        char const* address_file = std::getenv("RAPIDSMPF_ROOT_ADDRESS_FILE");
        if (address_file != nullptr) {
            auto listener_address = comm->listener_address();
            auto root_address_str =
                std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address)
                    ->getString();

            std::ofstream addr_file(address_file);
            if (!addr_file) {
                throw std::runtime_error(
                    "Failed to write root address to file: " + std::string{address_file}
                );
            }
            addr_file << root_address_str << std::endl;
            addr_file.close();
        }
    }

    return comm;
}
}  // namespace rapidsmpf::bootstrap

#endif  // RAPIDSMPF_HAVE_UCXX
