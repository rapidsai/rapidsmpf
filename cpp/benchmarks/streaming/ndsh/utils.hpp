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
[[nodiscard]] std::string get_table_path(
    std::string const& input_directory, std::string const& table_name
);
/*
Print the table to the logger.

Note that this requires RAPIDSMPF_LOG to be set to DEBUG or TRACE.

@param ctx The context.
@param table The table to print.
@param label The label to print.
*/
void debug_print_table(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    cudf::table_view const& table,
    std::string const& label
);
}  // namespace detail

[[nodiscard]] streaming::TableChunk to_device(
    std::shared_ptr<streaming::Context> ctx,
    streaming::TableChunk&& chunk,
    bool allow_overbooking = false
);

}  // namespace rapidsmpf::ndsh
