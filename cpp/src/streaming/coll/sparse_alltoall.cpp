/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/coll/sparse_alltoall.hpp>

namespace rapidsmpf::streaming {

SparseAlltoall::SparseAlltoall(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    std::vector<Rank> srcs,
    std::vector<Rank> dsts
)
    : ctx_{std::move(ctx)},
      exchange_{coll::SparseAlltoall(
          std::move(comm),
          op_id,
          ctx_->br().get(),
          std::move(srcs),
          std::move(dsts),
          [this]() {
              // Schedule waiters to resume on the executor.
              // This doesn't resume the frame immediately so we don't have to track
              // completion of this callback with a task_group.
              event_.set(ctx_->executor()->get());
          }
      )} {}

SparseAlltoall::~SparseAlltoall() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        event_.is_set(),
        "~SparseAlltoall: not all notification tasks complete, did you forget to await "
        "this->insert_finished()?"
    );
}

std::shared_ptr<Context> const& SparseAlltoall::ctx() const noexcept {
    return ctx_;
}

std::shared_ptr<Communicator> const& SparseAlltoall::comm() const noexcept {
    return exchange_.comm();
}

void SparseAlltoall::insert(Rank dst, PackedData&& packed_data) {
    exchange_.insert(dst, std::move(packed_data));
}

coro::task<void> SparseAlltoall::insert_finished() {
    exchange_.insert_finished();
    co_await event_;
}

std::vector<PackedData> SparseAlltoall::extract(Rank src) {
    return exchange_.extract(src);
}

}  // namespace rapidsmpf::streaming
