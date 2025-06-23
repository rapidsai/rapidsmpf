/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <condition_variable>  // NOLINT(unused-includes)
#include <mutex>  // NOLINT(unused-includes)

namespace rapidsmpf {


#define rapidsmpf_mutex_t std::mutex
#define rapidsmpf_condition_variable_t std::condition_variable


}  // namespace rapidsmpf
