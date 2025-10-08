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
 * @param semaphore Semaphore releasing resources to consumers.
 * @param latch Counting notifications so we can shutdown when all notifications have been
 * received.
 * @param ready_pids The ready set to insert the ready partition ID into.
 * @param pid The partition ID to insert.
 * @return A coroutine task that completes when the partition ID is inserted into the set.
 */
coro::task<void> insert_and_notify(
    std::mutex& mtx,
    coro::semaphore<std::numeric_limits<std::ptrdiff_t>::max()>& semaphore,
    coro::latch& latch,
    std::unordered_set<shuffler::PartID>& ready_pids,
    shuffler::PartID pid
) {
    // Note: this coroutine does not need to be scheduled, because it is offloaded to the
    // thread pool using a task_container.
    {
        std::unique_lock lock(mtx);
        RAPIDSMPF_EXPECTS(
            ready_pids.insert(pid).second,
            "something went wrong, pid is already in the ready set!"
        );
    }
    // keeping track of how many notifications we've received.
    latch.count_down();
    // Let a consumer know a pid is ready.
    co_await semaphore.release();
}

}  // namespace

ShufflerAsync::ShufflerAsync(
    std::shared_ptr<Context> ctx,
    OpID op_id,
    shuffler::PartID total_num_partitions,
    shuffler::Shuffler::PartitionOwner partition_owner
)
    : ctx_(std::move(ctx)),
      notifications_(ctx_->executor()),
      latch_{[&]() {
          // Need to initialise before shuffler_, so need to determine number of local
          // partitions sui generis.
          std::int64_t npart{0};
          for (shuffler::PartID i = 0; i < total_num_partitions; i++) {
              if (partition_owner(ctx_->comm(), i) == ctx_->comm()->rank()) {
                  npart++;
              }
          }
          return npart;
      }()},
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
                  notifications_.start(
                      insert_and_notify(mtx_, semaphore_, latch_, ready_pids_, pid)
                  ),
                  "failed to start task to notify waiters that the partition is ready"
              );
          },
          ctx_->statistics(),
          std::move(partition_owner)
      ) {
    RAPIDSMPF_EXPECTS(
        local_partitions().size() <= std::numeric_limits<std::int64_t>::max(),
        "Too many local partitions"
    );
}

ShufflerAsync::~ShufflerAsync() noexcept {
    if (!notifications_.empty()) {
        std::cerr << "~ShufflerAsync: not all notification tasks complete, remember to "
                     "await the finish token from this->insert_finished()"
                  << std::endl;
        std::terminate();
    }
    if (!ready_pids_.empty()) {
        ctx_->comm()->logger().warn("~ShufflerAsync: still ready partitions");
    }
    if (extracted_pids_.size() != shuffler_.local_partitions().size()) {
        ctx_->comm()->logger().warn(
            "~ShufflerAsync: not all partitions have been extracted"
        );
    }
}

std::span<shuffler::PartID const> ShufflerAsync::local_partitions() const {
    return shuffler_.local_partitions();
}

void ShufflerAsync::insert(std::unordered_map<shuffler::PartID, PackedData>&& chunks) {
    shuffler_.insert(std::move(chunks));
}

Node ShufflerAsync::insert_finished(std::vector<shuffler::PartID>&& pids) {
    shuffler_.insert_finished(std::move(pids));
    return finished_drain();
}

coro::task<std::optional<std::vector<PackedData>>> ShufflerAsync::extract_async(
    shuffler::PartID pid
) {
    // Ensure that `pid` is owned by this rank.
    RAPIDSMPF_EXPECTS(
        shuffler_.partition_owner(ctx_->comm(), pid) == ctx_->comm()->rank(),
        "the pid isn't owned by this rank, see ShufflerAsync::partition_owner()",
        std::out_of_range
    );

    while (true) {
        // We don't care if the semaphore is shut down here, our pid might still be in the
        // ready set to extract.
        std::ignore = co_await semaphore_.acquire();
        // Note: we cannot rely on RAII for unlocking in the "success" case because the
        // resumption of another coroutine might send us directly to a frame that needs
        // the lock.
        std::unique_lock lock(mtx_);
        if (extracted_pids_.contains(pid)) {
            lock.unlock();
            co_return std::nullopt;
        }
        if (ready_pids_.erase(pid) > 0) {
            RAPIDSMPF_EXPECTS(
                extracted_pids_.emplace(pid).second,
                "something went wrong, pid was both in the ready and the extracted set!"
            );
            lock.unlock();
            co_return shuffler_.extract(pid);
        }
        // The pid we are waiting on is not yet available, release so that someone else
        // can try and extract the pid that was available.
        lock.unlock();
        co_await semaphore_.release();
    }
}

coro::task<std::optional<ShufflerAsync::ExtractResult>>
ShufflerAsync::extract_any_async() {
    // We don't care if the semaphore is shut down here, there might still be pids in the
    // ready set to extract.
    std::ignore = co_await semaphore_.acquire();
    // Note: we cannot rely on RAII for unlocking in the "success" case because the
    // resumption of another coroutine might send us directly to a frame that needs
    // the lock.
    std::unique_lock lock(mtx_);
    // Did we wake up because a partition is ready?.
    if (!ready_pids_.empty()) {
        // Move a pid from the ready to the extracted set.
        auto pid = ready_pids_.extract(ready_pids_.begin()).value();
        RAPIDSMPF_EXPECTS(
            extracted_pids_.emplace(pid).second,
            "something went wrong, pid is already in the extracted set!"
        );
        lock.unlock();
        co_return std::make_pair(pid, shuffler_.extract(pid));
    }
    // If not, we were released because all partitions have been extracted.
    lock.unlock();
    co_return std::nullopt;
}

Node ShufflerAsync::finished_drain() {
    // Wait for all notifications to have fired
    co_await latch_;
    // Now wait for them to complete, otherwise coroutine frame unwinding can reach the
    // shuffler's destructor while the notification callback still references members.
    co_await notifications_.yield_until_empty();
    // And wake up any pending extraction tasks.
    co_await semaphore_.shutdown();
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
    co_await ctx->executor()->schedule();
    ShutdownAtExit c{ch_in, ch_out};
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
    auto finish_token = shuffler_async.insert_finished(std::move(finished));

    for ([[maybe_unused]] auto& _ : shuffler_async.local_partitions()) {
        auto finished = co_await shuffler_async.extract_any_async();
        RAPIDSMPF_EXPECTS(finished.has_value(), "extract_any_async returned null");

        co_await ch_out->send(
            std::make_unique<PartitionVectorChunk>(
                finished->first, std::move(finished->second)
            )
        );
    }
    co_await finish_token;
    co_await ch_out->drain(ctx->executor());
}

}  // namespace node
}  // namespace rapidsmpf::streaming
