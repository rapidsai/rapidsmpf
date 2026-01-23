/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/streaming/chunks/packed_data.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>

namespace rapidsmpf::streaming {

AllGather::AllGather(std::shared_ptr<Context> ctx, OpID op_id)
    : ctx_{std::move(ctx)},
      gatherer_{allgather::AllGather(
          ctx_->comm(),
          ctx_->progress_thread(),
          op_id,
          ctx_->br().get(),
          ctx_->statistics(),
          [this]() {
              // Schedule waiters to resume on the executor.
              // This doesn't resume the frame immediately so we don't have to track
              // completion of this callback with a task_group.
              event_.set(ctx_->executor()->get());
          }
      )} {}

AllGather::~AllGather() {
    if (!event_.is_set()) {
        std::cerr << "~AllGather: not all notification tasks complete, did you forget to "
                     "await this->extract_all() or to this->insert_finished()?"
                  << std::endl;
        std::terminate();
    }
}

[[nodiscard]] std::shared_ptr<Context> AllGather::ctx() const noexcept {
    return ctx_;
}

void AllGather::insert(std::uint64_t sequence_number, PackedData&& packed_data) {
    gatherer_.insert(sequence_number, std::move(packed_data));
}

void AllGather::insert_finished() {
    gatherer_.insert_finished();
}

coro::task<std::vector<PackedData>> AllGather::extract_all(AllGather::Ordered ordered) {
    // Wait until we're notified that everything is done.
    co_await event_;
    // And now this will not block.
    co_return gatherer_.wait_and_extract(ordered);
}

namespace node {
Node allgather(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    AllGather::Ordered ordered
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    auto gatherer = AllGather(ctx, op_id);
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        gatherer.insert(msg.sequence_number(), msg.release<PackedData>());
    }
    gatherer.insert_finished();
    auto data = co_await gatherer.extract_all(ordered);
    std::uint64_t sequence{0};
    for (auto&& chunk : data) {
        co_await ch_out->send(
            to_message(sequence++, std::make_unique<PackedData>(std::move(chunk)))
        );
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace node
}  // namespace rapidsmpf::streaming
