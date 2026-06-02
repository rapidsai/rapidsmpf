/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <memory>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/communicator/logger.hpp>
#include <rapidsmpf/progress_thread.hpp>

namespace rapidsmpf {

namespace ucxx {
class UCXX;
}

namespace bootstrap {

/**
 * @brief Create a UCXX communicator using the bootstrap backend.
 *
 * This function creates a fully initialized UCXX communicator by:
 * 1. Initializing the bootstrap context (rank, nranks)
 * 2. If rank 0: Creating UCXX root and publishing its address
 * 3. If rank != 0: Retrieving root address and connecting
 * 4. Performing a barrier to ensure all ranks are connected
 *
 * The function handles all coordination transparently based on the detected
 * or specified backend.
 *
 * @param progress_thread Progress thread for the initialized communicator.
 * @param type Backend to use.
 * @param options Configuration options for the UCXX communicator.
 * @param logger Externally provided logger. Must be non-null. The communicator
 * will overwrite the logger's rank to the bootstrapped UCXX rank.
 * @return Shared pointer to initialized UCXX communicator.
 * @throws std::runtime_error if initialization fails.
 *
 * @code
 * auto options = rapidsmpf::config::Options{};
 * auto progress = std::make_shared<rapidsmpf::ProgressThread>();
 * auto logger = std::make_shared<rapidsmpf::Logger>(options);
 * auto comm = rapidsmpf::bootstrap::create_ucxx_comm(
 *     progress, rapidsmpf::bootstrap::BackendType::AUTO, options, logger
 * );
 * comm->logger()->print("Hello from rank " + std::to_string(comm->rank()));
 * @endcode
 */
std::shared_ptr<ucxx::UCXX> create_ucxx_comm(
    std::shared_ptr<ProgressThread> progress_thread,
    BackendType type,
    config::Options options,
    std::shared_ptr<Logger> logger
);

}  // namespace bootstrap
}  // namespace rapidsmpf

#endif  // RAPIDSMPF_HAVE_UCXX
