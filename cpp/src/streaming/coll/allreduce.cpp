/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/coll/allreduce.hpp>

namespace rapidsmpf::streaming {

AllReduce::AllReduce(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Communicator> comm,
    std::unique_ptr<Buffer> input,
    std::unique_ptr<Buffer> output,
    OpID op_id,
    coll::ReduceOperator reduce_operator
)
    : ctx_{std::move(ctx)},
      reducer_{coll::AllReduce(
          std::move(comm),
          std::move(input),
          std::move(output),
          op_id,
          std::move(reduce_operator),
          [this]() {
              // Schedule waiters to resume on the executor.
              // This doesn't resume the frame immediately so we don't have to track
              // completion of this callback with a task_group.
              event_.set(ctx_->executor()->get());
          }
      )} {}

AllReduce::~AllReduce() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        event_.is_set(),
        "~AllReduce: not all notification tasks complete, did you forget to await "
        "this->extract()?"
    );
}

[[nodiscard]] std::shared_ptr<Context> const& AllReduce::ctx() const noexcept {
    return ctx_;
}

[[nodiscard]] std::shared_ptr<Communicator> const& AllReduce::comm() const noexcept {
    return reducer_.comm();
}

coro::task<std::pair<std::unique_ptr<Buffer>, std::unique_ptr<Buffer>>>
AllReduce::extract() {
    // Wait until we're notified that everything is done.
    co_await event_;
    // And now this will not block.
    co_return reducer_.wait_and_extract();
}

}  // namespace rapidsmpf::streaming
