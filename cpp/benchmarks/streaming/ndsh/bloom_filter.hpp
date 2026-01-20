/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <memory>

#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

#include "bloom_filter_impl.hpp"

namespace rapidsmpf::ndsh {

/**
 * @brief Build a bloom filter of the input channel.
 *
 * @param ctx Streaming context.
 * @param ch_in Input channel of `TableChunk`s to build bloom filter for.
 * @param ch_out Output channel receiving a single message containing the bloom filter.
 * @param tag Disambiguating tag to combine filters across ranks.
 * @param seed Hash seed for hashing the keys.
 * @param num_filter_blocks Number of blocks in the filter.
 *
 * @return Coroutine representing the construction of the bloom filter.
 */
[[maybe_unused]] streaming::Node build_bloom_filter(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    OpID tag,
    std::uint64_t seed,
    std::size_t num_filter_blocks
);

/**
 * @brief Apply a bloom filter to an input channel.
 *
 * @param ctx Streaming context.
 * @param bloom_filter Channel containing the bloom filter (a single message).
 * @param ch_in Input channel of `TableChunk`s to apply bloom filter to.
 * @param ch_out Output channel receiving filtered `TableChunk`s.
 * @param keys Indices selecting the key columns for the hash fingerprint
 *
 * @return Coroutine representing the application of the bloom filter.
 */
streaming::Node apply_bloom_filter(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> bloom_filter,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys
);

}  // namespace rapidsmpf::ndsh
