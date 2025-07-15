/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>

namespace rapidsmpf::shuffler::detail {

FinishCounter::FinishCounter(Rank nranks, std::vector<PartID> const& local_partitions)
    : nranks_{nranks} {
    // Initially, none of the partitions are ready to wait on.
    for (auto pid : local_partitions) {
        // partitions_ready_to_wait_on_.insert({pid, false});
        goalposts_.emplace(pid, PartitionInfo{0, 0, 0});
    }
}

bool FinishCounter::all_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return goalposts_.empty();
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
        cv_.notify_all();
    }
}

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

PartID FinishCounter::wait_any(std::optional<std::chrono::milliseconds> timeout) {
    PartID finished_key{std::numeric_limits<PartID>::max()};

    std::unique_lock<std::mutex> lock(mutex_);
    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&] {
        return goalposts_.empty()
               || std::ranges::any_of(
                   goalposts_,
                   [&, nranks = nranks_](auto const& item) {
                       auto done = item.second.is_finished(nranks);
                       if (done) {
                           finished_key = item.first;
                       }
                       return done;
                   }
               );
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
    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&, nranks = nranks_] {
        auto it = goalposts_.find(pid);
        RAPIDSMPF_EXPECTS(
            it != goalposts_.end(), "PartID has already been extracted", std::out_of_range
        );
        return it->second.is_finished(nranks);
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

    std::vector<PartID> result{};
    // TODO: hand-writing iteration rather than range-for to avoid
    // needing to rehash the key during extract_key. Needs
    // std::ranges, I think.
    for (auto it = goalposts_.begin(); it != goalposts_.end();) {
        // extract_key invalidates the iterator
        auto& [pid, p_info] = *it;
        if (p_info.is_finished(nranks_)) {
            result.push_back(pid);
            it = goalposts_.erase(it);
        } else {
            ++it;
        }
    }
    return result;
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
