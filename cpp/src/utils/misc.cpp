/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <ranges>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

#if __has_include(<valgrind/valgrind.h>)
#include <valgrind/valgrind.h>

bool is_running_under_valgrind() {
    static bool ret = RUNNING_ON_VALGRIND;
    return ret;
}
#else
bool is_running_under_valgrind() {
    return false;
}
#endif

}  // namespace rapidsmpf
