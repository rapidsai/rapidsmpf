/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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

/// @brief Handler to implement the wait* methods using callbacks
class FinishCounter::WaitHandler {
  public:
    WaitHandler() = default;

    ~WaitHandler() = default;

    /// @brief Callback to listen on the finished partitions
    void on_finished_cb(PartID pid) {
        {
            std::lock_guard lock(mutex);
            to_wait.emplace(pid);
        }
        cv.notify_all();
    }

    PartID wait_any(std::optional<std::chrono::milliseconds> timeout) {
        std::unique_lock lock(mutex);
        wait_for_if_timeout_else_wait(lock, cv, timeout, [&] {
            return !active || !to_wait.empty();
        });
        RAPIDSMPF_EXPECTS(active, "wait callback already finished", std::runtime_error);
        return to_wait.extract(to_wait.begin()).value();
    }

    void wait_on(PartID pid, std::optional<std::chrono::milliseconds> timeout) {
        std::unique_lock lock(mutex);
        wait_for_if_timeout_else_wait(lock, cv, timeout, [&] {
            return !active || to_wait.contains(pid);
        });
        RAPIDSMPF_EXPECTS(active, "wait callback already finished", std::runtime_error);
        to_wait.erase(pid);
    }

    std::unordered_set<PartID> to_wait{};  ///< finished partitions available to wait on
    bool active{true};
    std::condition_variable cv;
    std::mutex mutex;
};

FinishCounter::FinishCounter(
    Rank nranks,
    std::vector<PartID> const& local_partitions,
    std::function<void(PartID)>&& empty_partition_cb
)
    : nranks_{nranks},
      empty_partition_cb_{std::forward<std::function<void(PartID)>>(empty_partition_cb)},
      wait_handler_{std::make_unique<WaitHandler>()} {
    goalposts_.reserve(local_partitions.size());
    for (auto pid : local_partitions) {
        goalposts_.emplace(pid, PartitionInfo{});
    }
    finished_partitions_.reserve(local_partitions.size());

    register_finished_callback([wait_handler_ptr = wait_handler_.get()](PartID pid) {
        wait_handler_ptr->on_finished_cb(pid);
    });
}

FinishCounter::~FinishCounter() {
    // remove callback container before destroying the wait handler
    finished_cbs_.clear();
    wait_handler_.reset();
}

bool FinishCounter::all_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return finished_partitions_.size() == goalposts_.size();
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

        // Call empty partition callback if partition has no data, while holding the lock
        // (to prevent caller threads from extracting the empty partition)
        if (p_info.data_chunk_goal() == 0 && empty_partition_cb_) {
            empty_partition_cb_(pid);
        }
        lock.unlock();

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

    // capture the current size and iterate over existing elements without copying.
    auto curr_finished_pid_count = finished_partitions_.size();

    if (curr_finished_pid_count == goalposts_.size())
    {  // all partitions have already finished. So, finished_partitions_ vector would not
        // be updated further.
        lock.unlock();
        for (auto& pid : finished_partitions_) {
            cb(pid);
        }
        return invalid_cb_id;  // no need to register the callback
    }

    // while holding the lock, register the callback, so that any finished partitions will
    // be passed to the callback from the progress thread.
    auto cb_id = invalid_cb_id;
    FinishedCallback cb_copy = cb;
    {
        std::lock_guard cb_container_lock(finished_cbs_mutex_);
        cb_id = next_finished_cb_id_++;
        finished_cbs_.emplace_back(cb_id, curr_finished_pid_count, std::move(cb));
    }
    lock.unlock();

    // call the callback for each partition that has finished so far
    // Note: finished_partitions_ is already reserved and can only grow, never shrink or
    // reorder, so it's safe to iterate up to curr_finished_pid_count even after unlocking
    for (size_t i = 0; i < curr_finished_pid_count; ++i) {
        cb_copy(finished_partitions_[i]);
    }

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

PartID FinishCounter::wait_any(std::optional<std::chrono::milliseconds> timeout) {
    return wait_handler_->wait_any(std::move(timeout));
}

void FinishCounter::wait_on(
    PartID pid, std::optional<std::chrono::milliseconds> timeout
) {
    wait_handler_->wait_on(pid, std::move(timeout));
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
