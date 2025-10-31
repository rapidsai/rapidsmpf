/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <chrono>
#include <memory>

#include <cuda_device_runtime_api.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>

namespace rapidsmpf {

namespace bootstrap {

std::shared_ptr<ucxx::UCXX> create_ucxx_comm(Backend backend, config::Options options) {
    auto ctx = init(backend);

    // Ensure CUDA context is created before UCX is initialized
    cudaFree(nullptr);

    std::shared_ptr<ucxx::UCXX> comm;

    if (ctx.rank == 0) {
        // Create root UCXX communicator
        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);

        // Get the listener address and publish
        auto listener_address = comm->listener_address();
        auto root_worker_address_str =
            std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address)
                ->getString();
        put(ctx, "ucxx_root_address", root_worker_address_str);
    } else {
        // Worker ranks retrieve the root address and connect
        auto root_worker_address_str =
            get(ctx, "ucxx_root_address", std::chrono::milliseconds{30000});
        auto root_worker_address =
            ::ucxx::createAddressFromString(root_worker_address_str);

        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
    }

    comm->barrier();

    return comm;
}

}  // namespace bootstrap

}  // namespace rapidsmpf

#endif  // RAPIDSMPF_HAVE_UCXX
