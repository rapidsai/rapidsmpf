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
 * @brief Asynchronously pushes all messages from a vector into an output channel.
 *
 * Sends each message of the input vector into the channel in order,
 * marking the end of the stream once done.
 *
 * @param ctx The node context to use.
 * @param ch_out Output channel to which messages will be sent.
 * @param messages Input vector containing the messages to send.
 * @return Streaming node representing the asynchronous operation.
 *
 * @throws std::invalid_argument if any of the elements in messages is empty.
 */
Node push_to_channel(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::vector<Message> messages
);

/**
 * @brief Asynchronously pulls all messages from an input channel into a vector.
 *
 * Receives messages from the channel until it is closed and appends them
 * to the provided output vector.
 *
 * @param ctx The node context to use.
 * @param ch_in Input channel providing messages.
 * @param out_messages Output vector to store the received messages.
 * @return Streaming node representing the asynchronous operation.
 */
Node pull_from_channel(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::vector<Message>& out_messages
);

}  // namespace rapidsmpf::streaming::node
