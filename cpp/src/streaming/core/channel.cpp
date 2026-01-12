/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
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

Node Channel::drain(std::shared_ptr<CoroThreadPoolExecutor> executor) {
    return rb_.shutdown_drain(executor->get());
}

Node Channel::shutdown() {
    return rb_.shutdown();
}

bool Channel::empty() const noexcept {
    return rb_.empty();
}

bool Channel::is_shutdown() const noexcept {
    return rb_.is_shutdown();
}

}  // namespace rapidsmpf::streaming
