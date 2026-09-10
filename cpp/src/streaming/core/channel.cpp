/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

coro::task<bool> Channel::send(Message msg) {
    RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");
    metrics_.record_send(msg);
    auto result = co_await rb_.produce(sm_->insert(std::move(msg)));
    co_return result == coro::ring_buffer_result::produce::produced;
}

coro::task<Message> Channel::receive() {
    auto msg_id = co_await rb_.consume();
    if (msg_id.has_value()) {
        auto msg = sm_->extract(*msg_id);
        metrics_.record_receive(msg);
        co_return std::move(msg);
    } else {
        co_return Message{};
    }
}

void Channel::Metrics::record_send(Message const& msg) noexcept {
    message_count.fetch_add(1, std::memory_order_relaxed);
    auto const& cd = msg.content_description();
    spillable_count.fetch_add(
        static_cast<std::uint32_t>(cd.spillable()), std::memory_order_relaxed
    );
    for (auto mem_type : MEMORY_TYPES) {
        send_bytes[static_cast<std::size_t>(mem_type)].fetch_add(
            cd.content_size(mem_type), std::memory_order_relaxed
        );
    }
}

void Channel::Metrics::record_receive(Message const& msg) noexcept {
    auto const& cd = msg.content_description();
    for (auto mem_type : MEMORY_TYPES) {
        recv_bytes[static_cast<std::size_t>(mem_type)].fetch_add(
            cd.content_size(mem_type), std::memory_order_relaxed
        );
    }
}

Channel::MetricsSnapshot Channel::Metrics::snapshot() const noexcept {
    MetricsSnapshot snapshot;
    for (std::size_t i = 0; i < MEMORY_TYPES.size(); i++) {
        snapshot.send_bytes[i] = send_bytes[i].load(std::memory_order_relaxed);
        snapshot.recv_bytes[i] = recv_bytes[i].load(std::memory_order_relaxed);
    }
    snapshot.message_count = message_count.load(std::memory_order_relaxed);
    snapshot.spillable_count = spillable_count.load(std::memory_order_relaxed);
    return snapshot;
}

coro::task<bool> Channel::send_metadata(Message msg) {
    RAPIDSMPF_EXPECTS(!msg.empty(), "message cannot be empty");
    auto result = co_await metadata_.push(std::move(msg));
    co_return result == coro::queue_produce_result::produced;
}

Actor Channel::drain_metadata(std::shared_ptr<CoroThreadPoolExecutor> executor) {
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

Actor Channel::drain(std::shared_ptr<CoroThreadPoolExecutor> executor) {
    coro_results(
        co_await coro::when_all(
            rb_.shutdown_drain(executor->get()), drain_metadata(executor)
        )
    );
}

Actor Channel::shutdown() {
    coro_results(co_await coro::when_all(metadata_.shutdown(), rb_.shutdown()));
}

Actor Channel::shutdown_metadata() {
    return metadata_.shutdown();
}

bool Channel::empty() const noexcept {
    return rb_.empty();
}

bool Channel::is_shutdown() const noexcept {
    return rb_.is_shutdown();
}

Channel::MetricsSnapshot Channel::metrics() const noexcept {
    return metrics_.snapshot();
}
}  // namespace rapidsmpf::streaming
