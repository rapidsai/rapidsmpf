/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>

namespace rapidsmpf::streaming {

ShufflerAsync::ShufflerAsync(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
)
    : ctx_(std::move(ctx)),
      shuffler_(
          std::move(comm),
          op_id,
          total_num_partitions,
          ctx_->br().get(),
          [this]() {
              // Schedule waiters to resume on the executor.
              // This doesn't resume the frame immediately so we don't have to track
              // completion of this callback with a task_group.
              event_.set(ctx_->executor()->get());
          },
          std::move(partition_owner)
      ) {}

ShufflerAsync::~ShufflerAsync() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        event_.is_set(),
        "~ShufflerAsync: shuffle not complete, remember to await the "
        "finish token from this->insert_finished()"
    );
}

std::span<shuffler::PartID const> ShufflerAsync::local_partitions() const {
    return shuffler_.local_partitions();
}

void ShufflerAsync::insert(std::unordered_map<shuffler::PartID, PackedData>&& chunks) {
    shuffler_.insert(std::move(chunks));
}

Actor ShufflerAsync::insert_finished() {
    shuffler_.insert_finished();
    co_await event_;
}

std::vector<PackedData> ShufflerAsync::extract(shuffler::PartID pid) {
    return shuffler_.extract(pid);
}

namespace actor {

Actor shuffler(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
) {
    co_await ctx->executor()->schedule();
    ShutdownAtExit c{ch_in, ch_out};
    ShufflerAsync shuffler_async(
        ctx, std::move(comm), op_id, total_num_partitions, std::move(partition_owner)
    );

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto partition_map = msg.release<PartitionMapChunk>();
        shuffler_async.insert(std::move(partition_map.data));
    }

    co_await shuffler_async.insert_finished();

    for (auto pid : shuffler_async.local_partitions()) {
        auto chunks = shuffler_async.extract(pid);
        co_await ch_out->send(to_message(
            pid,
            std::make_unique<PartitionVectorChunk>(
                PartitionVectorChunk{.data = std::move(chunks)}
            )
        ));
    }
    co_await ch_out->drain(ctx->executor());
}

}  // namespace actor
}  // namespace rapidsmpf::streaming
