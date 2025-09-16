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


namespace {

coro::task<void> insert_to_set_and_notify(
    coro::mutex& mtx,
    coro::condition_variable& cv,
    std::unordered_set<shuffler::PartID>& set,
    shuffler::PartID pid
) {
    // Note: this coroutine is not needed to be scheduled, because it is called from the
    // progress thread.
    {
        auto lock = co_await mtx.scoped_lock();
        set.insert(pid);
    }
    co_await cv.notify_all();
}

}  // namespace

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
              coro::sync_wait(insert_to_set_and_notify(mtx_, cv_, ready_pids_, pid));
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

    auto chunks = shuffler_.extract(pid);
    // shuffler gets marked as finished when the partitions are extracted. So, tasks
    // waiting on the cv should be notified.
    if (shuffler_.finished()) {
        co_await cv_.notify_all();
    }

    co_return std::move(chunks);
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

    auto chunks = shuffler_.extract(pid);
    // shuffler gets marked as finished when the partitions are extracted. So, tasks
    // waiting on the cv should be notified.
    if (shuffler_.finished()) {
        co_await cv_.notify_all();
    }

    co_return {pid, std::move(chunks)};
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
    ShufflerAsync shuffler_async(
        ctx, stream, op_id, total_num_partitions, std::move(partition_owner)
    );

    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();
    CudaEvent event;

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto partition_map = msg.release<PartitionMapChunk>();

        // Make sure that the input chunk's stream is in sync with shuffler's stream.
        utils::sync_streams(stream, partition_map.stream, event);

        shuffler_async.insert(std::move(partition_map.data));
    }

    // Tell the shuffler that we have no more input data.
    std::vector<rapidsmpf::shuffler::PartID> finished(
        shuffler_async.total_num_partitions()
    );
    std::iota(finished.begin(), finished.end(), 0);
    shuffler_async.insert_finished(std::move(finished));

    while (!shuffler_async.finished()) {
        auto result = co_await shuffler_async.extract_any_async();
        if (!result.is_valid()) {
            break;
        }

        co_await ch_out->send(
            std::make_unique<PartitionVectorChunk>(result.pid, std::move(result.chunks))
        );
    }

    co_await ch_out->drain(ctx->executor());
}

}  // namespace node
}  // namespace rapidsmpf::streaming
