/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <memory>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::ndsh {

enum class ConcatOrder : bool {
    DONT_CARE,
    LINEARIZE,
};

streaming::Node concatenate(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    ConcatOrder order = ConcatOrder::DONT_CARE
);

}  // namespace rapidsmpf::ndsh
