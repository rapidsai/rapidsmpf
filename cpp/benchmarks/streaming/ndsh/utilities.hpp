/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <memory>
#include <string>
#include <vector>

#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::ndsh {
namespace detail {
[[nodiscard]] std::vector<std::string> list_parquet_files(std::string const& root_path);
}

[[nodiscard]] streaming::TableChunk to_device(
    std::shared_ptr<streaming::Context> ctx, streaming::TableChunk&& chunk
);
}  // namespace rapidsmpf::ndsh
