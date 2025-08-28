/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <utility>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>

namespace rapidsmpf::shuffler::detail {

FinishCounter::FinishCounter(
    Rank nranks,
    std::vector<PartID> const& local_partitions,
    std::function<void(PartID)>&& empty_partition_cb
)
    : nranks_{nranks},
      n_local_partitions_{local_partitions.size()},
      empty_partition_cb_{std::forward<std::function<void(PartID)>>(empty_partition_cb)} {
    goalposts_.reserve(n_local_partitions_);
    for (auto pid : local_partitions) {
        goalposts_.emplace(pid, PartitionInfo{});
    }
    finished_cbs_.reserve(n_local_partitions_);
}

bool FinishCounter::all_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return goalposts_.empty() || finished_partitions_.size() == n_local_partitions_;
}

void FinishCounter::move_goalpost(PartID pid, ChunkID nchunks) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto& p_info = goalposts_[pid];
    p_info.move_goalpost(nchunks, nranks_);
}

void FinishCounter::add_finished_chunk(PartID pid) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto& p_info = goalposts_[pid];

    p_info.add_finished_chunk(nranks_);

    if (p_info.is_finished(nranks_)) {
        finished_partitions_.emplace_back(pid);
        bool is_empty = p_info.data_chunk_goal() == 0;
        lock.unlock();  // no longer need the lock

        // Call empty partition callback if partition has no data
        if (is_empty && empty_partition_cb_) {
            empty_partition_cb_(pid);
        }

        cv_.notify_all();  // notify any waiting threads

        std::lock_guard cb_container_lock(finished_cbs_mutex_);
        // Note: finished_partitions_ does not need to be protected by the lock
        // because it is only appended by the progress thread.
        for (auto& cb_container : finished_cbs_) {
            while (cb_container.next_pid_idx < finished_partitions_.size()) {
                cb_container.cb(finished_partitions_[cb_container.next_pid_idx++]);
            }
        }
    }
}

FinishCounter::FinishedCbId FinishCounter::register_finished_callback(
    FinishedCallback&& cb
) {
    std::unique_lock lock(mutex_);

    if (finished_partitions_.size() == n_local_partitions_)
    {  // all partitions have already finished. So, finished_partitions_ vector would not
        // be updated further.
        lock.unlock();
        for (auto& pid : finished_partitions_) {
            cb(pid);
        }
        return invalid_cb_id;  // no need to register the callback
    }

    // take a copy of the finished_partitions_ vector and unlock the mutex
    std::vector<PartID> finished_partitions = finished_partitions_;
    auto curr_finished_pid_count = finished_partitions_.size();
    lock.unlock();

    // call the callback for each partition that has finished
    for (auto& pid : finished_partitions) {
        cb(pid);
    }

    std::lock_guard cb_container_lock(finished_cbs_mutex_);
    auto cb_id = next_finished_cb_id_++;
    finished_cbs_.emplace_back(cb_id, curr_finished_pid_count, std::move(cb));
    return cb_id;
}

void FinishCounter::remove_finished_callback(FinishedCbId cb_id) {
    if (cb_id != invalid_cb_id) {
        std::lock_guard cb_container_lock(finished_cbs_mutex_);
        std::erase_if(finished_cbs_, [cb_id](auto& cb_container) {
            return cb_container.cb_id == cb_id;
        });
    }
}

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

PartID FinishCounter::wait_any(std::optional<std::chrono::milliseconds> timeout) {
    PartID finished_key{std::numeric_limits<PartID>::max()};

    std::unique_lock<std::mutex> lock(mutex_);
    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&] {
        return goalposts_.empty()
               || std::ranges::any_of(goalposts_, [&](auto const& item) {
                      auto done = item.second.is_finished(nranks_);
                      if (done) {
                          finished_key = item.first;
                      }
                      return done;
                  });
    });

    RAPIDSMPF_EXPECTS(
        finished_key != std::numeric_limits<PartID>::max(),
        "no more partitions to wait on",
        std::out_of_range
    );

    // We extract the partition to avoid returning the same partition twice.
    goalposts_.erase(finished_key);
    return finished_key;
}

void FinishCounter::wait_on(
    PartID pid, std::optional<std::chrono::milliseconds> timeout
) {
    std::unique_lock<std::mutex> lock(mutex_);
    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&] {
        auto it = goalposts_.find(pid);
        RAPIDSMPF_EXPECTS(
            it != goalposts_.end(), "PartID has already been extracted", std::out_of_range
        );
        return it->second.is_finished(nranks_);
    });
    goalposts_.erase(pid);
}

std::vector<PartID> FinishCounter::wait_some(
    std::optional<std::chrono::milliseconds> timeout
) {
    std::unique_lock<std::mutex> lock(mutex_);
    RAPIDSMPF_EXPECTS(
        !goalposts_.empty(), "no more partitions to wait on", std::out_of_range
    );

    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&]() {
        return std::ranges::any_of(goalposts_, [nranks = nranks_](auto const& item) {
            return item.second.is_finished(nranks);
        });
    });

    std::vector<PartID> pids{};
    for (auto it = goalposts_.begin(); it != goalposts_.end();) {
        auto& [pid, p_info] = *it;
        if (p_info.is_finished(nranks_)) {
            pids.push_back(pid);
            it = goalposts_.erase(it);
        } else {
            ++it;
        }
    }

    return pids;
}

std::string detail::FinishCounter::str() const {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "FinishCounter(goalposts={";
    for (auto const& [pid, p_info] : goalposts_) {
        ss << "p" << pid << ": (rank_count= " << p_info.rank_count
           << ", chunk_goal= " << p_info.chunk_goal
           << ", finished_chunk_count= " << p_info.finished_chunk_count
           << ", is_finished= " << p_info.is_finished(nranks_) << "), ";
    }
    ss << (goalposts_.empty() ? "}" : "\b\b}") << ")";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
