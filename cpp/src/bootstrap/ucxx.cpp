/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>


#ifdef RAPIDSMPF_HAVE_UCXX

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include <cuda_device_runtime_api.h>
#include <unistd.h>  // for unsetenv

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/ucxx.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/progress_thread.hpp>

namespace rapidsmpf::bootstrap {

namespace {
// Hex encoding for binary-safe address transmission
std::string hex_encode(std::string_view input) {
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

}  // namespace

std::shared_ptr<ucxx::UCXX> create_ucxx_comm(
    std::shared_ptr<ProgressThread> progress_thread,
    BackendType type,
    config::Options options
) {
    auto ctx = init(type);

    // Ensure CUDA context is created before UCX is initialized
    cudaFree(nullptr);

    std::shared_ptr<ucxx::UCXX> comm;

    auto address_file = getenv_optional("RRUN_ROOT_ADDRESS_FILE");

    if (ctx.rank == 0) {
        // Root rank: create listener and publish address for other ranks.
        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, std::nullopt, options);
        comm = std::make_shared<ucxx::UCXX>(
            std::move(ucxx_initialized_rank), options, progress_thread
        );

        auto listener_address = comm->listener_address();
        auto root_worker_address =
            std::get<std::shared_ptr<::ucxx::Address>>(listener_address.address);

        // Publish via the bootstrap backend (FileBackend or SlurmBackend) so
        // that local non-root ranks can retrieve it with get().
        put(ctx, "ucxx_root_address", root_worker_address->getStringView());

        // In Slurm hybrid mode the parent rrun process also needs the address
        // to relay it to parents on other nodes via PMIx.  Write a hex-encoded
        // copy to the address file for the parent to pick up.
        if (address_file.has_value()) {
            std::string encoded_address =
                hex_encode(root_worker_address->getStringView());
            std::string const temp_path = *address_file + ".tmp";
            std::ofstream addr_ofs(temp_path);
            if (!addr_ofs) {
                throw std::runtime_error(
                    "Failed to write root address to file: " + temp_path
                );
            }
            addr_ofs << encoded_address << std::endl;
            addr_ofs.close();
            if (std::rename(temp_path.c_str(), address_file->c_str()) != 0) {
                std::remove(temp_path.c_str());
                throw std::runtime_error(
                    "Failed to rename root address file to: " + *address_file
                );
            }

            unsetenv("RRUN_ROOT_ADDRESS_FILE");
        }

        sync(ctx);
    } else {
        // Non-root ranks: retrieve the root address and connect.
        sync(ctx);

        auto root_worker_address_str =
            get(ctx, "ucxx_root_address", std::chrono::seconds{30});
        auto root_worker_address =
            ::ucxx::createAddressFromString(root_worker_address_str);

        auto ucxx_initialized_rank =
            ucxx::init(nullptr, ctx.nranks, root_worker_address, options);
        comm = std::make_shared<ucxx::UCXX>(
            std::move(ucxx_initialized_rank), options, progress_thread
        );
    }

    comm->barrier();

    return comm;
}
}  // namespace rapidsmpf::bootstrap

#endif  // RAPIDSMPF_HAVE_UCXX
