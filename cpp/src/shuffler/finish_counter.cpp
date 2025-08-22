/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>

namespace rapidsmpf::shuffler::detail {

// CallbackGuard template implementation
template <typename T>
    requires std::same_as<T, PartID> || std::same_as<T, FinishCounter::FinishedCbId>
struct FinishCounter::CallbackGuard {
    friend class FinishCounter;

    // Delete copy and move operations to ensure unique ownership
    CallbackGuard(const CallbackGuard&) = delete;
    CallbackGuard& operator=(const CallbackGuard&) = delete;
    CallbackGuard(CallbackGuard&&) = delete;
    CallbackGuard& operator=(CallbackGuard&&) = delete;

    ~CallbackGuard() {
        if constexpr (std::same_as<T, FinishCounter::FinishedCbId>) {
            fc_.cancel_finished_any_callback(cb_id_);
        } else {
            fc_.cancel_finished_callback(cb_id_);
        }
    }

  private:
    CallbackGuard(FinishCounter& fc, T cb_id) : fc_(fc), cb_id_(cb_id) {}

    FinishCounter& fc_;
    T cb_id_;
};

// Explicit template instantiations
template struct FinishCounter::CallbackGuard<PartID>;
template struct FinishCounter::CallbackGuard<FinishCounter::FinishedCbId>;

// Function definitions that use CallbackGuard
FinishCounter::CallbackGuard<PartID> FinishCounter::on_finished_with_guard(
    PartID pid, FinishedCallback&& cb
) {
    on_finished(pid, std::move(cb));
    return CallbackGuard<PartID>(*this, pid);
}

FinishCounter::CallbackGuard<FinishCounter::FinishedCbId>
FinishCounter::on_finished_any_with_guard(FinishedAnyCallback&& cb) {
    auto cb_id = on_finished_any(std::move(cb));
    return CallbackGuard<FinishedCbId>(*this, cb_id);
}

FinishCounter::FinishCounter(Rank nranks, std::vector<PartID> const& local_partitions)
    : nranks_{nranks} {
    // Initially, none of the partitions are ready to wait on.
    goalposts_.reserve(local_partitions.size());
    for (auto pid : local_partitions) {
        goalposts_.emplace(pid, PartitionInfo{});
    }
}

bool FinishCounter::all_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return goalposts_.empty() && ready_pids_stash_.empty();
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
        // remove the partition from the goalposts_ map
        auto nh = goalposts_.extract(pid);

        auto& extracted_p_info = nh.mapped();
        bool has_data = extracted_p_info.data_chunk_goal() != 0;

        if (extracted_p_info.finished_cb != nullptr) {
            // FinishedCallback already registered for this pid. Notify and release it.
            lock.unlock();
            extracted_p_info.finished_cb(has_data);
        } else if (!finished_any_cbs_.empty()) {
            // there are some FinishedAnyCallbacks available. Notify the one in FIFO
            // order.
            auto first_nh = finished_any_cbs_.extract(finished_any_cbs_.begin());
            lock.unlock();

            first_nh.mapped()(pid, has_data);
        } else {
            // no callbacks registered. Add the pid to the ready_pids_ map. (still locked)
            RAPIDSMPF_EXPECTS(
                ready_pids_stash_.emplace(pid, has_data).second,
                "pid already in ready_pids_stash_"
            );
        }
    }
}

FinishCounter::FinishedCbId FinishCounter::on_finished_any(FinishedAnyCallback&& cb) {
    std::unique_lock lock(mutex_);
    // if there are any ready pids in the stash, notify the callback with the first pid
    // and remove it from the stash
    if (!ready_pids_stash_.empty()) {
        auto nh = ready_pids_stash_.extract(ready_pids_stash_.begin());
        lock.unlock();
        cb(nh.key(), nh.mapped());
        return invalid_cb_id;
    } else if (!goalposts_.empty()) {
        // no ready pids in the stash, but there are some partitions in the goalposts_ map
        // which will be finished later. add the callback to the queue
        auto cb_id = next_finished_cb_id_++;
        finished_any_cbs_.emplace(cb_id, std::move(cb));
        return cb_id;
    } else {
        // no ready pids in the stash, and no partitions in the goalposts_ map. raise an
        // error
        RAPIDSMPF_FAIL("All partitions are finished");
    }
}

void FinishCounter::cancel_finished_any_callback(FinishedCbId cb_id) {
    if (cb_id != invalid_cb_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_any_cbs_.erase(cb_id);
    }
}

void FinishCounter::on_finished(PartID pid, FinishedCallback&& cb) {
    std::unique_lock lock(mutex_);

    // cb will be evaluated in the following order:
    // 1. if pid is in the goalposts_ map, check if the callback is already registered
    // 2. if pid is in the ready_pids_stash_ map, call the callback immediately
    // 3. if pid is not in the goalposts_ map or the ready_pids_stash_ map, raise an error

    auto it = goalposts_.find(pid);
    if (it != goalposts_.end()) {
        // pid is in the goalposts_ map. check if the callback is already registered
        auto& p_info = it->second;
        if (p_info.finished_cb) {
            RAPIDSMPF_FAIL(
                "callback already registered for partition " + std::to_string(pid)
            );
        } else if (p_info.is_finished(nranks_)) {
            // partition is already finished (unlikely case), call the callback
            // immediately by extracting the partition from the goalposts_ map
            auto extracted_nh = goalposts_.extract(it);
            auto& extracted_p_info = extracted_nh.mapped();
            lock.unlock();

            cb(extracted_p_info.data_chunk_goal() != 0);
        } else {  // register the callback
            p_info.finished_cb = std::move(cb);
        }
    } else if (auto it1 = ready_pids_stash_.find(pid); it1 != ready_pids_stash_.end()) {
        // pid is in the ready_pids_stash_ map. call the callback immediately
        auto nh = ready_pids_stash_.extract(it1);
        lock.unlock();

        cb(nh.mapped());
    } else {
        // pid is not in the goalposts_ map or the ready_pids_stash_ map. raise an error
        RAPIDSMPF_FAIL("Partition already finished " + std::to_string(pid));
    }
}

void FinishCounter::cancel_finished_callback(PartID pid) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = goalposts_.find(pid);
    if (it != goalposts_.end()) {
        auto& p_info = it->second;
        p_info.finished_cb = nullptr;
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

std::pair<PartID, bool> FinishCounter::wait_any(
    std::optional<std::chrono::milliseconds> timeout
) {
    std::pair<PartID, bool> result{std::numeric_limits<PartID>::max(), false};
    bool finished{false};
    auto cb_guard = on_finished_any_with_guard(
        [this, &result, &finished](PartID cb_run_pid, bool cb_run_has_data) {
            {
                {
                    std::lock_guard<std::mutex> lock(this->wait_mutex_);
                    result.first = cb_run_pid;
                    result.second = cb_run_has_data;
                    finished = true;
                }
                this->wait_cv_.notify_one();
            }
        }
    );

    std::unique_lock<std::mutex> lock(this->wait_mutex_);
    wait_for_if_timeout_else_wait(lock, this->wait_cv_, timeout, [&finished] {
        return finished;
    });

    RAPIDSMPF_EXPECTS(
        result.first != std::numeric_limits<PartID>::max(),
        "no partition finished",
        std::runtime_error
    );

    return result;
}

bool FinishCounter::wait_on(
    PartID pid, std::optional<std::chrono::milliseconds> timeout
) {
    bool has_data{false}, finished{false};
    auto cb_guard =
        on_finished_with_guard(pid, [this, &has_data, &finished](bool cb_run_has_data) {
            {
                std::lock_guard<std::mutex> lock(this->wait_mutex_);
                has_data = cb_run_has_data;
                finished = true;
            }
            this->wait_cv_.notify_one();
        });

    std::unique_lock<std::mutex> lock(this->wait_mutex_);
    wait_for_if_timeout_else_wait(lock, this->wait_cv_, timeout, [&finished] {
        return finished;
    });
    return has_data;
}

std::pair<std::vector<PartID>, std::vector<bool>> FinishCounter::wait_some(
    std::optional<std::chrono::milliseconds> /* timeout */
) {
    // std::unique_lock<std::mutex> lock(mutex_);
    // RAPIDSMPF_EXPECTS(
    //     !goalposts_.empty(), "no more partitions to wait on", std::out_of_range
    // );

    // wait_for_if_timeout_else_wait(lock, cv_, timeout, [&]() {
    //     return std::ranges::any_of(goalposts_, [nranks = nranks_](auto const& item) {
    //         return item.second.is_finished(nranks);
    //     });
    // });

    // std::vector<PartID> pids{};
    // std::vector<bool> contains_data{};
    // for (auto it = goalposts_.begin(); it != goalposts_.end();) {
    //     auto& [pid, p_info] = *it;
    //     if (p_info.is_finished(nranks_)) {
    //         pids.push_back(pid);
    //         contains_data.push_back(p_info.data_chunk_goal() != 0);
    //         it = goalposts_.erase(it);
    //     } else {
    //         ++it;
    //     }
    // }
    // return {std::move(pids), std::move(contains_data)};
    return {};
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
