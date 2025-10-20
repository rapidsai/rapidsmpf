/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming::node {

/**
 * @brief Fanout policy controlling how messages are propagated.
 */
enum class FanoutPolicy : int {
    /// @brief Process messages as they arrive and immediately forward them.
    BOUNDED,
    /// @brief Accumulate all incoming messages before forwarding them.
    UNBOUNDED,
};

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * The node continuously receives messages from the input channel and forwards
 * them to all output channels according to the selected policy:
 *
 * - `FanoutPolicy::BOUNDED`: Messages are forwarded as they arrive.
 *   The next message is only broadcast once *all* output channels have finished
 *   sending the current one. This provides backpressure so slow consumers
 *   naturally throttle the upstream flow, but it can cause head-of-line
 *   blocking and even deadlock if downstream rates differ.
 *
 * - `FanoutPolicy::UNBOUNDED`: Messages are broadcast to all output channels
 *   with no backpressure (potentially unbounded memory usage), allowing
 *   downstream consumers to process at independent rates.
 *
 * Each output channel receives a shallow copy of the same message; no payload
 * data is duplicated. All copies share the same underlying payload, ensuring
 * zero-copy broadcast semantics.
 *
 * @param ctx The node context to use.
 * @param ch_in Input channel from which messages are received.
 * @param chs_out Output channels to which messages are broadcast.
 * @param policy Fanout strategy (`BOUNDED` or `UNBOUNDED`).
 *
 * @return Streaming node representing the fanout operation.
 *
 * @throws std::invalid_argument If an unknown fanout policy is specified.
 *
 * @note Since messages are shallow-copied, releasing a payload (`release<T>()`)
 * is only valid on messages that hold exclusive ownership of the payload.
 */
Node fanout(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<std::shared_ptr<Channel>> chs_out,
    FanoutPolicy policy
);

}  // namespace rapidsmpf::streaming::node
