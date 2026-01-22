/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

coro::task<bool> Channel::send(Message msg) {
    RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");
    auto result = co_await rb_.produce(sm_->insert(std::move(msg)));
    co_return result == coro::ring_buffer_result::produce::produced;
}

coro::task<Message> Channel::receive() {
    auto msg_id = co_await rb_.consume();
    if (msg_id.has_value()) {
        co_return sm_->extract(*msg_id);
    } else {
        co_return Message{};
    }
}

coro::task<bool> Channel::send_metadata(Message msg) {
    RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");
    auto result = co_await metadata_.push(std::move(msg));
    co_return result == coro::queue_produce_result::produced;
}

Node Channel::drain_metadata(std::shared_ptr<CoroThreadPoolExecutor> executor) {
    return metadata_.shutdown_drain(executor->get());
}

coro::task<Message> Channel::receive_metadata() {
    auto msg = co_await metadata_.pop();
    if (msg.has_value()) {
        co_return std::move(*msg);
    } else {
        co_return Message{};
    }
}

Node Channel::drain(std::shared_ptr<CoroThreadPoolExecutor> executor) {
    coro_results(
        co_await coro::when_all(
            rb_.shutdown_drain(executor->get()), drain_metadata(executor)
        )
    );
}

Node Channel::shutdown() {
    coro_results(co_await coro::when_all(metadata_.shutdown(), rb_.shutdown()));
}

Node Channel::shutdown_metadata() {
    return metadata_.shutdown();
}

bool Channel::empty() const noexcept {
    return rb_.empty();
}

bool Channel::is_shutdown() const noexcept {
    return rb_.is_shutdown();
}

}  // namespace rapidsmpf::streaming
