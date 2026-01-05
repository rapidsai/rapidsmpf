/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

#include "test_allreduce_custom_type.hpp"

using rapidsmpf::MemoryType;
using rapidsmpf::PackedData;
using rapidsmpf::coll::ReduceOperator;
using rapidsmpf::coll::ReduceOperatorType;

ReduceOperator make_custom_value_reduce_operator_device() {
    return rapidsmpf::coll::detail::make_device_reduce_operator<CustomValue>(
        [] __device__(CustomValue a, CustomValue b) {
            CustomValue out{};
            out.value = a.value + b.value;
            out.weight = a.weight < b.weight ? a.weight : b.weight;
            return out;
        }
    );
}
