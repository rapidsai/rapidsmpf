/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::ndsh {
namespace detail {
std::vector<std::string> list_parquet_files(std::string const& root_path) {
    // Files are named `ANYTHING_somenumber.ANYTHING.parquet` Should be sorted in
    // ascending order by their numerical part. root_path is the path to the directory
    // containing the files.

    std::vector<std::string> result;
    if (std::filesystem::directory_entry(std::filesystem::path(root_path))
            .is_regular_file())
    {
        result.push_back(root_path);
    } else {
        std::vector<std::pair<int, std::string>> files;

        for (const auto& entry : std::filesystem::directory_iterator(root_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();

                if (filename.ends_with(".parquet")) {
                    // Look for pattern: ANYTHING_number.ANYTHING.parquet
                    // Find the last underscore before a number
                    std::size_t underscore_pos = std::string::npos;
                    std::size_t number_start = std::string::npos;
                    std::size_t number_end = std::string::npos;

                    // Search for underscore followed by digits
                    for (std::size_t i = 0; i < filename.length(); ++i) {
                        if (filename[i] == '_' && i + 1 < filename.length()
                            && ::isdigit(filename[i + 1]))
                        {
                            // Found underscore followed by digit, check if all following
                            // chars until dot are digits
                            std::size_t j = i + 1;
                            while (j < filename.length() && ::isdigit(filename[j])) {
                                ++j;
                            }
                            // Check if we found digits followed by a dot (start of
                            // extension part)
                            if (j > i + 1 && j < filename.length() && filename[j] == '.')
                            {
                                underscore_pos = i;
                                number_start = i + 1;
                                number_end = j;
                            }
                        }
                    }

                    if (underscore_pos != std::string::npos
                        && number_start != std::string::npos
                        && number_end != std::string::npos)
                    {
                        std::string number =
                            filename.substr(number_start, number_end - number_start);
                        if (!number.empty() && std::ranges::all_of(number, ::isdigit)) {
                            files.emplace_back(std::stoi(number), entry.path().string());
                        }
                    }
                }
            }
        }
        std::ranges::sort(files, std::less{}, [](auto const& pair) {
            return pair.first;
        });
        result.reserve(files.size());
        for (auto const& [_, path] : files) {
            result.push_back(path);
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
