/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

std::ostream& operator<<(std::ostream& os, MemoryType mem_type) {
    return os << to_string(mem_type);
}

std::istream& operator>>(std::istream& is, MemoryType& out) {
    std::string token;
    if (!(is >> token)) {
        return is;
    }
    token = to_upper(token);
    if (token == "DEVICE") {
        out = MemoryType::DEVICE;
    } else if (token == "PINNED_HOST" || token == "PINNED" || token == "PINNED-HOST") {
        out = MemoryType::PINNED_HOST;
    } else if (token == "HOST") {
        out = MemoryType::HOST;
    } else {
        is.setstate(std::ios::failbit);
    }
    return is;
}
}  // namespace rapidsmpf
