/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::ndsh {
namespace detail {
std::vector<std::string> list_parquet_files(std::string const& root_path) {
    // Files are named `ANYTHING_somenumber.ANYTHING.parquet` Should be sorted in
    // ascending order by their numerical part. root_path is the path to the directory
    // containing the files.

    auto root_entry = std::filesystem::directory_entry(std::filesystem::path(root_path));
    RAPIDSMPF_EXPECTS(
        root_entry.exists()
            && (root_entry.is_regular_file() || root_entry.is_directory()),
        "Invalid file path",
        std::runtime_error
    );
    if (root_entry.is_regular_file()) {
        RAPIDSMPF_EXPECTS(
            root_path.ends_with(".parquet"), "Invalid filename", std::runtime_error
        );
        return {root_path};
    }
    std::vector<std::string> result;
    for (auto const& entry : std::filesystem::directory_iterator(root_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.ends_with(".parquet")) {
                result.push_back(entry.path());
            }
        }
    }
    return result;
}

std::string get_table_path(
    std::string const& input_directory, std::string const& table_name
) {
    auto dir = input_directory.empty() ? "." : input_directory;
    auto file_path = dir + "/" + table_name + ".parquet";
    if (std::filesystem::exists(file_path)) {
        return file_path;
    }
    return dir + "/" + table_name + "/";
}

void debug_print_table(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    cudf::table_view const& table,
    std::string const& label
) {
    if (table.num_rows() == 0) {
        ctx->comm()->logger().debug("[DEBUG] ", label, " is empty");
        return;
    }
    ctx->comm()->logger().debug("[DEBUG] ", label, " rows ", table.num_rows());

    // For simplicity, just print that we have the table
    // To actually print values would require type dispatch and host copies
    for (cudf::size_type col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
        ctx->comm()->logger().debug(
            "  Column ",
            col_idx,
            ": type=",
            cudf::type_to_name(table.column(col_idx).type()),
            " size=",
            table.column(col_idx).size(),
            " nulls=",
            table.column(col_idx).null_count()
        );
    }
}


}  // namespace detail

streaming::TableChunk to_device(
    std::shared_ptr<streaming::Context> ctx,
    streaming::TableChunk&& chunk,
    bool allow_overbooking
) {
    auto reservation = ctx->br()->reserve_device_memory_and_spill(
        chunk.make_available_cost(), allow_overbooking
    );
    return chunk.make_available(reservation);
}
}  // namespace rapidsmpf::ndsh
