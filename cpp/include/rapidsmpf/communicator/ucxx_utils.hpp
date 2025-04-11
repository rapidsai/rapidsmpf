/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <rapidsmpf/communicator/ucxx.hpp>

namespace rapidsmpf {

namespace ucxx {

/**
 * @brief Initialize UCXX Communicator using MPI.
 *
 * @param mpi_comm MPI communicator.
 * @return UCXX communicator.
 *
 * @note Requires MPI to be initialized prior to calling this function.
 */
std::shared_ptr<UCXX> init_using_mpi(MPI_Comm mpi_comm);

}  // namespace ucxx
}  // namespace rapidsmpf
