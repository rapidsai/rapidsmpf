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
enum class FanoutPolicy : uint8_t {
    /**
     * @brief Process messages as they arrive and immediately forward them.
     *
     * Messages are forwarded as soon as they are received from the input channel.
     * The next message is not processed until all output channels have completed
     * sending the current one, ensuring backpressure and synchronized flow.
     */
    BOUNDED,

    /**
     * @brief Forward messages without enforcing backpressure.
     *
     * In this mode, messages may be accumulated internally before being
     * broadcast, or they may be forwarded immediately depending on the
     * implementation and downstream consumption rate.
     *
     * This mode disables coordinated backpressure between outputs, allowing
     * consumers to process at independent rates, but can lead to unbounded
     * buffering and increased memory usage.
     *
     * @note Consumers might not receive any messages until *all* upstream
     * messages have been sent, depending on the implementation and buffering
     * strategy.
     */
    UNBOUNDED,
};

/**
 * @brief Broadcast messages from one input channel to multiple output channels.
 *
 * The node continuously receives messages from the input channel and forwards
 * them to all output channels according to the selected fanout policy, see
 * ::FanoutPolicy.
 *
 * Each output channel receives a shallow copy of the same message; no payload
 * data is duplicated. All copies share the same underlying payload, ensuring
 * zero-copy broadcast semantics.
 *
 * @param ctx The node context to use.
 * @param ch_in Input channel from which messages are received.
 * @param chs_out Output channels to which messages are broadcast.
 * @param policy The fanout strategy to use (see ::FanoutPolicy).
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
