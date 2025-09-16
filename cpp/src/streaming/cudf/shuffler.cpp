/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <memory>
#include <numeric>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>
#include <rapidsmpf/streaming/cudf/utils.hpp>

namespace rapidsmpf::streaming {

ShufflerAsync::ShufflerAsync(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
)
    : ctx_(std::move(ctx)),
      shuffler_(
          ctx_->comm(),
          ctx_->progress_thread(),
          op_id,
          total_num_partitions,
          stream,
          ctx_->br(),
          [this](shuffler::PartID pid) -> void {
              ctx_->comm()->logger().trace("inserting finished partition ", pid);
              {
                  auto lock = mtx_.scoped_lock();
                  ready_pids_.insert(pid);
              }
              // notify cv using the progress thread when running the cb
              coro::sync_wait(cv_.notify_all());
          },
          ctx_->statistics(),
          std::move(partition_owner)
      ) {}

bool ShufflerAsync::finished() const {
    return shuffler_.finished();
}

void ShufflerAsync::insert(std::unordered_map<shuffler::PartID, PackedData>&& chunks) {
    shuffler_.insert(std::move(chunks));
}

void ShufflerAsync::insert_finished(std::vector<shuffler::PartID>&& pids) {
    shuffler_.insert_finished(std::move(pids));
}

coro::task<std::vector<PackedData>> ShufflerAsync::extract_async(shuffler::PartID pid) {
    // Wait until the partition is finished
    auto lock = co_await mtx_.scoped_lock();
    co_await cv_.wait(lock, [this, pid]() {
        return shuffler_.finished() || ready_pids_.contains(pid);
    });

    // partition not found (may have been already extracted or shuffler was finished
    // before the pid was inserted into ready_pids_)
    RAPIDSMPF_EXPECTS(
        ready_pids_.erase(pid) > 0,
        "partition ID not found: " + std::to_string(pid),
        std::runtime_error
    );
    lock.unlock();  // no longer need the lock

    ctx_->comm()->logger().trace("extracting finished partition ", pid);
    co_return shuffler_.extract(pid);
}

coro::task<ShufflerAsync::ExtractResult> ShufflerAsync::extract_any_async() {
    // wait until at least one partition is ready for extraction
    auto lock = co_await mtx_.scoped_lock();
    co_await cv_.wait(lock, [this]() {
        return shuffler_.finished() || !ready_pids_.empty();
    });

    // no partitions to extract or shuffle is already finished
    if (ready_pids_.empty()) {
        lock.unlock();
        ctx_->comm()->logger().warn("no partitions to extract");
        co_return ExtractResult::invalid();
    }

    auto pid = ready_pids_.extract(ready_pids_.begin()).value();
    lock.unlock();

    ctx_->comm()->logger().trace("extracting any finished partition ", pid);
    co_return {pid, shuffler_.extract(pid)};
}

namespace node {

Node shuffler(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
) {
    auto shuffler = std::make_shared<ShufflerAsync>(
        std::move(ctx), stream, op_id, total_num_partitions, std::move(partition_owner)
    );

    co_await coro::when_all(
        shuffler_async_insert(shuffler, stream, std::move(ch_in)),
        shuffler_async_extract(std::move(shuffler), std::move(ch_out))
    );
}

Node shuffler_async_insert(
    std::shared_ptr<ShufflerAsync> shuffler,
    rmm::cuda_stream_view stream,
    std::shared_ptr<Channel> ch_in
) {
    ShutdownAtExit c{ch_in};
    co_await shuffler->ctx()->executor()->schedule();
    CudaEvent event;

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto partition_map = msg.release<PartitionMapChunk>();

        // Make sure that the input chunk's stream is in sync with shuffler's stream.
        utils::sync_streams(stream, partition_map.stream, event);

        shuffler->insert(std::move(partition_map.data));
    }

    // Tell the shuffler that we have no more input data.
    std::vector<rapidsmpf::shuffler::PartID> finished(shuffler->total_num_partitions());
    std::iota(finished.begin(), finished.end(), 0);
    shuffler->insert_finished(std::move(finished));
    co_return;
}

Node shuffler_async_extract(
    std::shared_ptr<ShufflerAsync> shuffler, std::shared_ptr<Channel> ch_out
) {
    ShutdownAtExit c{ch_out};
    co_await shuffler->ctx()->executor()->schedule();

    while (!shuffler->finished()) {
        auto result = co_await shuffler->extract_any_async();
        if (!result.is_valid()) {
            break;
        }

        co_await ch_out->send(
            std::make_unique<PartitionVectorChunk>(result.pid, std::move(result.chunks))
        );
    }

    co_await ch_out->drain(shuffler->ctx()->executor());
}

}  // namespace node
}  // namespace rapidsmpf::streaming
