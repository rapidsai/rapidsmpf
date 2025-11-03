/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_UCXX

#include <memory>

#include <rapidsmpf/bootstrap/bootstrap.hpp>

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
 * @param backend Backend to use (default: AUTO for auto-detection).
 * @param options Configuration options for the UCXX communicator.
 * @return Shared pointer to initialized UCXX communicator.
 * @throws std::runtime_error if initialization fails.
 *
 * @code
 * auto comm = rapidsmpf::bootstrap::create_ucxx_comm();
 * comm->logger().print("Hello from rank " + std::to_string(comm->rank()));
 * @endcode
 */
std::shared_ptr<ucxx::UCXX> create_ucxx_comm(
    Backend backend = Backend::AUTO, config::Options options = config::Options{}
);

}  // namespace bootstrap
}  // namespace rapidsmpf

#endif  // RAPIDSMPF_HAVE_UCXX
