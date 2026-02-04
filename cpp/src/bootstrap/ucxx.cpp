/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include <cuda_device_runtime_api.h>
#include <unistd.h>  // for unsetenv

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::bootstrap {

namespace {
// Hex encoding for binary-safe address transmission
std::string hex_encode(std::string const& input) {
    static constexpr const char* hex_chars = "0123456789abcdef";
    std::string result;
    result.reserve(input.size() * 2);
    for (char ch : input) {
        auto c = static_cast<unsigned char>(ch);
        result.push_back(hex_chars[c >> 4]);
        result.push_back(hex_chars[c & 0x0F]);
    }
    return result;
}

std::string hex_decode(std::string const& input) {
    std::string result;
    result.reserve(input.size() / 2);
    for (size_t i = 0; i < input.size(); i += 2) {
        auto high = static_cast<unsigned char>(
            (input[i] >= 'a') ? (input[i] - 'a' + 10) : (input[i] - '0')
        );
        auto low = static_cast<unsigned char>(
            (input[i + 1] >= 'a') ? (input[i + 1] - 'a' + 10) : (input[i + 1] - '0')
        );
        result.push_back(static_cast<char>((high << 4) | low));
    }
    return result;
}
}  // namespace

std::shared_ptr<ucxx::UCXX> create_ucxx_comm(Backend backend, config::Options options) {
    auto ctx = init(backend);

    // Ensure CUDA context is created before UCX is initialized
    cudaFree(nullptr);

    std::shared_ptr<ucxx::UCXX> comm;

    auto precomputed_address_encoded = getenv_optional("RAPIDSMPF_ROOT_ADDRESS");
    auto address_file = getenv_optional("RAPIDSMPF_ROOT_ADDRESS_FILE");

    // Path 1: Early address mode for root rank in Slurm hybrid mode.
    // Rank 0 is launched first to create its address and write it to a file.
    // Parent will coordinate with other parents via PMIx, then launch worker ranks
    // with RAPIDSMPF_ROOT_ADDRESS set. No PMIx put/barrier/get bootstrap coordination.
    if (ctx.rank == 0 && address_file.has_value()) {
        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);

        auto listener_address = comm->listener_address();
        auto root_worker_address_str =
            std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address)
                ->getString();

        std::string encoded_address = hex_encode(root_worker_address_str);
        std::ofstream addr_file(*address_file);
        if (!addr_file) {
            throw std::runtime_error(
                "Failed to write root address to file: " + *address_file
            );
        }
        addr_file << encoded_address << std::endl;
        addr_file.close();

        auto verbose = getenv_optional("RAPIDSMPF_VERBOSE");
        if (verbose && *verbose == "1") {
            std::cerr << "[rank 0] Wrote address to " << *address_file
                      << ", skipping bootstrap coordination" << std::endl;
        }

        // Unset the flag so rank 0 participates in the final barrier
        unsetenv("RAPIDSMPF_ROOT_ADDRESS_FILE");
    }
    // Path 2: Slurm hybrid mode for non-root ranks.
    // Parent process already coordinated the root address via PMIx and provided it
    // via RAPIDSMPF_ROOT_ADDRESS environment variable (hex-encoded).
    else if (precomputed_address_encoded.has_value() && ctx.rank != 0)
    {
        std::string precomputed_address = hex_decode(*precomputed_address_encoded);
        auto root_worker_address = ::ucxx::createAddressFromString(precomputed_address);
        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
    }
    // Path 3: Normal bootstrap mode for root rank.
    // Create listener and publish address via put() for non-root ranks to retrieve.
    else if (ctx.rank == 0)
    {
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
    // Path 4: Normal bootstrap mode for non-root ranks.
    // Retrieve root address via get() and connect.
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
