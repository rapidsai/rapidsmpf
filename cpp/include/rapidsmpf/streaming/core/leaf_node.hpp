/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming::node {


// TODO: move to .cpp
inline Node push_to_channel(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel2> ch_out,
    std::vector<Message> messages
) {
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();

    for (auto& msg : messages) {
        RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty", std::invalid_argument);
        co_await ch_out->send(std::move(msg));
    }
    co_await ch_out->drain(ctx->executor());
}

inline Node pull_from_channel(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel2> ch_in,
    std::vector<Message>& out_messages
) {
    ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();

    while (true) {
        auto msg = co_await ch_in->receive();
        std::cout << "pull_from_channel()" << std::endl;
        if (msg.empty()) {
            break;
        }
        out_messages.push_back(std::move(msg));
    }
}

/**
 * @brief Asynchronously pushes all chunks from a vector into an output channel.
 *
 * Sends each element of the input vector into the channel in order,
 * marking the end of the stream once done.
 *
 * @tparam ChunkT The type of chunk being sent.
 * @param ctx The context to use.
 * @param ch_out Output channel to which chunks will be sent.
 * @param chunks Input vector containing the chunks to send.
 * @return Streaming node representing the asynchronous operation.
 */
template <typename ChunkT>
Node push_chunks_to_channel(
    std::shared_ptr<Context> ctx,
    SharedChannel<ChunkT> ch_out,
    std::vector<std::unique_ptr<ChunkT>> chunks
) {
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();

    for (auto& chunk : chunks) {
        RAPIDSMPF_EXPECTS(
            chunk, "`chunks` cannot contain null pointers", std::invalid_argument
        );
        co_await ch_out->send(std::move(chunk));
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Asynchronously pulls all chunks from an input channel into a vector.
 *
 * Receives elements from the channel until it is closed and appends them
 * to the provided output vector.
 *
 * @tparam ChunkT The type of chunk being collected.
 * @param ctx The context to use.
 * @param ch_in Input channel providing chunks.
 * @param output Output vector to store the received chunks.
 * @return Streaming node representing the asynchronous operation.
 */
template <typename ChunkT>
Node pull_chunks_from_channel(
    std::shared_ptr<Context> ctx,
    SharedChannel<ChunkT> ch_in,
    std::vector<std::unique_ptr<ChunkT>>& output
) {
    ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();

    while (true) {
        auto chunk = co_await ch_in->receive_or(nullptr);
        if (chunk == nullptr) {
            break;
        }
        output.push_back(std::move(chunk));
    }
}

}  // namespace rapidsmpf::streaming::node
