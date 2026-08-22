/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <rapidsmpf/cupti.hpp>

static_assert(
    std::is_same_v<
        decltype(std::declval<rapidsmpf::CuptiMonitor const&>().get_callback_counters()),
        std::unordered_map<rapidsmpf::CuptiCallbackId, std::size_t>>
);

int main() {
    return 0;
}
