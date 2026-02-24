/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <mpi.h>

#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/progress_thread.hpp>

namespace rapidsmpf {

namespace ucxx {

/**
 * @brief Initialize UCXX Communicator using MPI.
 *
 * @param mpi_comm MPI communicator.
 * @param options Configuration options.
 * @param progress_thread Progress thread for the initialized communicator.
 * @return UCXX communicator.
 *
 * @note Requires MPI to be initialized prior to calling this function.
 */
std::shared_ptr<UCXX> init_using_mpi(
    MPI_Comm mpi_comm,
    rapidsmpf::config::Options options,
    std::shared_ptr<ProgressThread> progress_thread
);

}  // namespace ucxx
}  // namespace rapidsmpf
