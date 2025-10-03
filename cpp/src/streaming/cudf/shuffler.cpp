/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <numeric>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/streaming/cudf/shuffler.hpp>

namespace rapidsmpf::streaming {

namespace {

/**
 * @brief Inserts a partition ID into a ready set and notifies all waiting tasks.
 *
 * @param mtx The mutex to use for synchronization.
 * @param cv The condition variable to use for notification.
 * @param ready_pids The ready set to insert the ready partition ID into.
 * @param pid The partition ID to insert.
 * @return A coroutine task that completes when the partition ID is inserted into the set.
 */
coro::task<void> insert_and_notify(
    coro::mutex& mtx,
    coro::condition_variable& cv,
    std::unordered_set<shuffler::PartID>& ready_pids,
    shuffler::PartID pid
) {
    // Note: this coroutine does not need to be scheduled, because it is offloaded to the
    // thread pool using `spawn`.
    {
        auto lock = co_await mtx.scoped_lock();
        RAPIDSMPF_EXPECTS(
            ready_pids.insert(pid).second,
            "something went wrong, pid is already in the ready set!"
        );
    }
    co_await cv.notify_all();
}

}  // namespace

ShufflerAsync::ShufflerAsync(
    std::shared_ptr<Context> ctx,
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
          ctx_->br(),
          [this](shuffler::PartID pid) -> void {
              ctx_->comm()->logger().trace("notifying waiters that ", pid, " is ready");
              // Libcoro may resume suspended coroutines during cv notification, using the
              // caller thread. Submitting a detached task ensures that the progress
              // thread is not used to resume the coroutines.
              RAPIDSMPF_EXPECTS(
                  ctx_->executor()->spawn(insert_and_notify(mtx_, cv_, ready_pids_, pid)),
                  "failed to spawn task to notify waiters that the partition is ready"
              );
          },
          ctx_->statistics(),
          std::move(partition_owner)
      ) {}

std::span<shuffler::PartID const> ShufflerAsync::local_partitions() const {
    return shuffler_.local_partitions();
}

void ShufflerAsync::insert(std::unordered_map<shuffler::PartID, PackedData>&& chunks) {
    shuffler_.insert(std::move(chunks));
}

void ShufflerAsync::insert_finished(std::vector<shuffler::PartID>&& pids) {
    shuffler_.insert_finished(std::move(pids));
}

coro::task<std::optional<std::vector<PackedData>>> ShufflerAsync::extract_async(
    shuffler::PartID pid
) {
    auto lock = co_await mtx_.scoped_lock();

    // Ensure that `pid` is owned by this rank.
    {
        auto pids = shuffler_.local_partitions();
        RAPIDSMPF_EXPECTS(
            std::ranges::find(pids, pid) != pids.end(),
            "the pid isn't owned by this rank, see ShufflerAsync::partition_owner()",
            std::out_of_range
        );
    }

    // Wait until the partition is ready or has been extracted (by somebody else).
    co_await cv_.wait(lock, [this, pid]() {
        return ready_pids_.contains(pid) || extracted_pids_.contains(pid);
    });

    // Did we wake up because the partition is ready?.
    if (ready_pids_.contains(pid)) {
        // pid is now being extracted and isn't ready anymore.
        RAPIDSMPF_EXPECTS(ready_pids_.erase(pid) > 0, "something went wrong");
        RAPIDSMPF_EXPECTS(extracted_pids_.emplace(pid).second, "something went wrong");
        co_return shuffler_.extract(pid);
    }
    // If not, we were woken because the partition was extracted by somebody else.
    co_return std::nullopt;
}

coro::task<std::optional<ShufflerAsync::ExtractResult>>
ShufflerAsync::extract_any_async() {
    auto const total_num_pids = shuffler_.local_partitions().size();
    auto lock = co_await mtx_.scoped_lock();

    // Wait until either all partitions has been extracted or at least one partition is
    // ready for extraction.
    co_await cv_.wait(lock, [this, total_num_pids]() {
        return extracted_pids_.size() == total_num_pids || !ready_pids_.empty();
    });

    // Did we wake up because a partition is ready?.
    if (!ready_pids_.empty()) {
        // Move a pid from the ready to the extracted set.
        auto pid = ready_pids_.extract(ready_pids_.begin()).value();
        RAPIDSMPF_EXPECTS(
            extracted_pids_.emplace(pid).second,
            "something went wrong, pid is already in the extracted set!"
        );
        co_return std::make_pair(pid, shuffler_.extract(pid));
    }
    // If not, we were woken because all partitions have been extracted.
    co_return std::nullopt;
}

namespace node {

Node shuffler(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
) {
    ShutdownAtExit c{ch_in, ch_out};
    co_await ctx->executor()->schedule();

    ShufflerAsync shuffler_async(
        ctx, op_id, total_num_partitions, std::move(partition_owner)
    );

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto partition_map = msg.release<PartitionMapChunk>();
        shuffler_async.insert(std::move(partition_map.data));
    }

    // Tell the shuffler that we have no more input data.
    std::vector<rapidsmpf::shuffler::PartID> finished(
        shuffler_async.total_num_partitions()
    );
    std::iota(finished.begin(), finished.end(), 0);
    shuffler_async.insert_finished(std::move(finished));

    for ([[maybe_unused]] auto& _ : shuffler_async.local_partitions()) {
        auto finished = co_await shuffler_async.extract_any_async();
        RAPIDSMPF_EXPECTS(finished.has_value(), "extract_any_async returned null");

        co_await ch_out->send(
            std::make_unique<PartitionVectorChunk>(
                finished->first, std::move(finished->second)
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

}  // namespace node
}  // namespace rapidsmpf::streaming
