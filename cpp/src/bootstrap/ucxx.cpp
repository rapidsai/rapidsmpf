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

    // Check if root address was provided by parent process (rrun hybrid mode)
    char const* precomputed_address_encoded = std::getenv("RAPIDSMPF_ROOT_ADDRESS");

    if (precomputed_address_encoded != nullptr && ctx.rank != 0) {
        // Parent process already coordinated the root address via PMIx
        // Address is hex-encoded to avoid issues with binary data in env vars
        // Note: Only non-root ranks use this path. Rank 0 should always create the
        // listener.
        std::string precomputed_address = hex_decode(precomputed_address_encoded);

        // Worker children connect using provided address
        auto root_worker_address = ::ucxx::createAddressFromString(precomputed_address);
        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
        comm = std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);
    } else {
        // Standard bootstrap coordination via put/get/barrier

        // Special case: If rank 0 is asked to write address file before full bootstrap,
        // it means we're in rrun hybrid parent-mediated mode where rank 0 is launched
        // first to get its address, then other ranks are launched later.
        // In this case, skip the put/barrier/get dance and just create the listener.
        char const* address_file = std::getenv("RAPIDSMPF_ROOT_ADDRESS_FILE");
        bool early_address_mode = (ctx.rank == 0 && address_file != nullptr);

        if (ctx.rank == 0) {
            // Create root UCXX communicator
            auto ucxx_initialized_rank =
                ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
            comm =
                std::make_shared<ucxx::UCXX>(std::move(ucxx_initialized_rank), options);

            // Get the listener address
            auto listener_address = comm->listener_address();
            auto root_worker_address_str =
                std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address)
                    ->getString();

            if (early_address_mode) {
                // Write address file immediately and skip bootstrap coordination
                // Parent will coordinate with other parents via PMIx
                // Encode as hex to avoid issues with binary data
                std::string encoded_address = hex_encode(root_worker_address_str);
                std::ofstream addr_file(address_file);
                if (!addr_file) {
                    throw std::runtime_error(
                        "Failed to write root address to file: "
                        + std::string{address_file}
                    );
                }
                addr_file << encoded_address << std::endl;
                addr_file.close();

                char const* verbose = std::getenv("RAPIDSMPF_VERBOSE");
                if (verbose && std::string{verbose} == "1") {
                    std::cerr << "[rank 0] Wrote address to " << address_file
                              << ", skipping bootstrap coordination" << std::endl;
                }

                // Unset the flag so rank 0 won't skip the final barrier
                // (we need all ranks to synchronize at the end)
                unsetenv("RAPIDSMPF_ROOT_ADDRESS_FILE");

                // Skip put/barrier - other ranks will get address via
                // RAPIDSMPF_ROOT_ADDRESS Return early, don't do full bootstrap
            } else {
                // Normal mode: publish address for other ranks
                put(ctx, "ucxx_root_address", root_worker_address_str);
            }
        }

        if (!early_address_mode) {
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
                comm = std::make_shared<ucxx::UCXX>(
                    std::move(ucxx_initialized_rank), options
                );
            }
        }
    }

    // Final barrier to synchronize all ranks before returning
    // Note: rank 0 in early address mode unsets RAPIDSMPF_ROOT_ADDRESS_FILE
    // after writing the file, so it participates in this barrier
    comm->barrier();

    return comm;
}
}  // namespace rapidsmpf::bootstrap

#endif  // RAPIDSMPF_HAVE_UCXX
