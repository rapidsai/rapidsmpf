/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>

#include <cuda_device_runtime_api.h>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::bootstrap {

bool is_running_with_rrun() {
    return std::getenv("RAPIDSMPF_RANK") != nullptr;
}

Rank get_nranks() {
    RAPIDSMPF_EXPECTS(
        is_running_with_rrun(),
        "get_nranks() can only be called when running with `rrun`. "
        "Set RAPIDSMPF_RANK environment variable or use a launcher like 'rrun'.",
        std::runtime_error
    );

    char const* nranks_str = std::getenv("RAPIDSMPF_NRANKS");
    RAPIDSMPF_EXPECTS(
        nranks_str != nullptr,
        "RAPIDSMPF_NRANKS environment variable not set. "
        "Make sure to use a rrun launcher to call this function.",
        std::runtime_error
    );

    try {
        return std::stoi(nranks_str);
    } catch (...) {
        RAPIDSMPF_FAIL(
            "Failed to parse integer from RAPIDSMPF_NRANKS environment variable: "
                + std::string(nranks_str),
            std::runtime_error
        );
    }
}

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
