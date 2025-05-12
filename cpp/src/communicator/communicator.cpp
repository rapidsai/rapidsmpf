/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/communicator/communicator.hpp>

namespace rapidsmpf {

Communicator::Logger::Logger(Communicator* comm)
    : comm_{comm}, level_{level_from_env()} {};

}  // namespace rapidsmpf
