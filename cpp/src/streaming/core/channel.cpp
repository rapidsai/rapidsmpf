/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/core/channel.hpp>

namespace rapidsmpf::streaming {

coro::task<bool> Channel::send(Message msg) {
    auto result = co_await rb_.produce(std::move(msg));
    co_return result == coro::ring_buffer_result::produce::produced;
}

coro::task<Message> Channel::receive() {
    auto msg = co_await rb_.consume();
    if (msg.has_value()) {
        co_return std::move(*msg);
    } else {
        co_return Message{};
    }
}

Node Channel::drain(std::unique_ptr<coro::thread_pool>& executor) {
    return rb_.shutdown_drain(executor);
}

Node Channel::shutdown() {
    return rb_.shutdown();
}

bool Channel::empty() const noexcept {
    return rb_.empty();
}

}  // namespace rapidsmpf::streaming
