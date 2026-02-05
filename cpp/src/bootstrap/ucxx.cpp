/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <chrono>
#include <memory>
#include <string>

#include <cuda_device_runtime_api.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::bootstrap {

std::shared_ptr<ucxx::UCXX> create_ucxx_comm(BackendType type, config::Options options) {
    auto ctx = init(type);

    // Ensure CUDA context is created before UCX is initialized
    cudaFree(nullptr);

    std::shared_ptr<ucxx::UCXX> comm;

    // Root rank: Create listener and publish address via put() for non-root ranks.
    if (ctx.rank == 0) {
        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);

        auto listener_address = comm->listener_address();
        auto root_worker_address_str =
            std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address)
                ->getString();

        put(ctx, "ucxx_root_address", root_worker_address_str);
        sync(ctx);
    }
    // Non-root ranks: Retrieve root address via get() and connect.
    else
    {
        sync(ctx);

        auto root_worker_address_str =
            get(ctx, "ucxx_root_address", std::chrono::seconds{30});
        auto root_worker_address =
            ::ucxx::createAddressFromString(root_worker_address_str);

        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
    }

    comm->barrier();

    return comm;
}
}  // namespace rapidsmpf::bootstrap

#endif  // RAPIDSMPF_HAVE_UCXX
