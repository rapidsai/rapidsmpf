/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include <cudf/types.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::ndsh {
enum class KeepKeys : bool {
    NO,
    YES,
};

streaming::Node inner_join_broadcast(
    std::shared_ptr<streaming::Context> ctx,
    // We will always choose left as build table and do "broadcast" joins
    std::shared_ptr<streaming::Channel> left,
    std::shared_ptr<streaming::Channel> right,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on,
    OpID tag,
    KeepKeys keep_keys = KeepKeys::YES
);
streaming::Node inner_join_shuffle(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> left,
    std::shared_ptr<streaming::Channel> right,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on,
    KeepKeys keep_keys = KeepKeys::YES
);
streaming::Node shuffle(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    std::vector<cudf::size_type> keys,
    std::uint32_t num_partitions,
    OpID tag
);
}  // namespace rapidsmpf::ndsh
