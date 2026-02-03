/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/coll/allreduce.hpp>

/**
 * @brief Simple user-defined type used to demonstrate custom allreduce operators.
 *
 * The reduction we implement for this type is:
 *  - `value` field: summed across ranks
 *  - `weight` field: minimum across ranks
 */
struct CustomValue {
    int value{};
    int weight{};
};

/**
 * @brief Factory for a device-side reduction operator for `CustomValue`.
 *
 * The returned operator expects `PackedData::data` to contain a contiguous array
 * of `CustomValue` in device memory, with equal sizes for all ranks.
 */
rapidsmpf::coll::ReduceOperator make_custom_value_reduce_operator_device();
