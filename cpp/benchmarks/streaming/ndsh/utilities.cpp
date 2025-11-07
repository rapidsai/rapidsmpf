/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/error.hpp>
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
    for (const auto& entry : std::filesystem::directory_iterator(root_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.ends_with(".parquet")) {
                result.push_back(entry.path().string());
            }
        }
    }
    return result;
}

}  // namespace detail

streaming::TableChunk to_device(
    std::shared_ptr<streaming::Context> ctx, streaming::TableChunk&& chunk
) {
    auto reservation = ctx->br()->reserve_and_spill(
        MemoryType::DEVICE, chunk.make_available_cost(), false
    );
    return chunk.make_available(reservation);
}
}  // namespace rapidsmpf::ndsh
