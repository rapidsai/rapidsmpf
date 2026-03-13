/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>

namespace rapidsmpf::shuffler::detail {

namespace {
/**
 * @brief A utility function to wait on a predicate with a timeout and throw if
 * the timeout is reached. If the timeout is not set, wait for the predicate
 * to be true.
 *
 * @tparam Pred The type of the predicate.
 *
 * @param lock The lock to use for the wait.
 * @param cv The condition variable to use for the wait.
 * @param timeout The timeout to use for the wait.
 * @param pred The predicate to wait on.
 *
 * @throws std::runtime_error if the timeout is reached.
 */
template <typename Pred>
void wait_for_if_timeout_else_wait(
    std::unique_lock<std::mutex>& lock,
    std::condition_variable& cv,
    std::optional<std::chrono::milliseconds>& timeout,
    Pred&& pred
) {
    if (timeout.has_value()) {
        // if the timeout is set, and pred() is not true, throw
        RAPIDSMPF_EXPECTS(
            cv.wait_for(lock, *timeout, std::move(pred)),
            "wait timeout reached",
            std::runtime_error
        );
    } else {
        cv.wait(lock, std::move(pred));
    }
}

}  // namespace

FinishCounter::FinishCounter(
    Rank nranks,
    std::span<PartID const> local_partitions,
    FinishedCallback&& finished_callback
)
    : nranks_{nranks},
      n_unfinished_partitions_{safe_cast<PartID>(local_partitions.size())},
      rank_reported_(safe_cast<std::size_t>(nranks), false),
      local_partitions_(local_partitions),
      finished_callback_{std::forward<FinishedCallback>(finished_callback)} {}

bool FinishCounter::all_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return n_unfinished_partitions_ == 0;
}

void FinishCounter::move_goalpost(Rank src_rank, ChunkID nchunks) {
    std::unique_lock<std::mutex> lock(mutex_);
    RAPIDSMPF_EXPECTS(nchunks != 0, "the goalpost was moved by 0 chunks");
    RAPIDSMPF_EXPECTS(
        !rank_reported_[safe_cast<std::size_t>(src_rank)],
        "the goalpost was moved more than once for the same rank"
    );
    rank_reported_[safe_cast<std::size_t>(src_rank)] = true;
    n_ranks_with_goalpost_++;
    total_chunk_goal_ += nchunks;
}

void FinishCounter::add_finished_chunk() {
    std::unique_lock<std::mutex> lock(mutex_);
    total_finished_chunks_++;
    RAPIDSMPF_EXPECTS(
        (n_ranks_with_goalpost_ < nranks_)
            || (total_finished_chunks_ <= total_chunk_goal_),
        "finished chunks exceed the goal"
    );

    if (n_ranks_with_goalpost_ == nranks_ && total_finished_chunks_ == total_chunk_goal_
        && !all_done_)
    {
        all_done_ = true;
        n_unfinished_partitions_ = 0;
        lock.unlock();

        wait_cv_.notify_all();  // notify any waiting threads

        if (finished_callback_) {  // notify the callback for each partition
            for (auto pid : local_partitions_) {
                finished_callback_(pid);
            }
        }
    }
}

void FinishCounter::wait(std::optional<std::chrono::milliseconds> timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    wait_for_if_timeout_else_wait(lock, wait_cv_, timeout, [&] { return all_done_; });
}

std::string detail::FinishCounter::str() const {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "FinishCounter(n_ranks_with_goalpost=" << n_ranks_with_goalpost_
       << ", total_chunk_goal=" << total_chunk_goal_
       << ", total_finished_chunks=" << total_finished_chunks_
       << ", all_done=" << all_done_ << ")";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
