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
        partitions_ready_to_wait_on_.insert({pid, false});
    }
}

void FinishCounter::move_goalpost(PartID pid, ChunkID nchunks) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto& [rank_counter, chunk_goal] = goalposts_[pid];
    RAPIDSMPF_EXPECTS(
        rank_counter++ < nranks_, "the goalpost was moved more than one per rank"
    );
    chunk_goal += nchunks;
}

void FinishCounter::add_finished_chunk(PartID pid) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto& finished_chunk = ++finished_chunk_counters_[pid];
    auto& [rank_counter, chunk_goal] = goalposts_[pid];

    // The partition is finished if the goalpost has been set by all ranks
    // and the number of finished chunks has reach the goal.
    if (rank_counter == nranks_) {
        if (finished_chunk == chunk_goal) {
            partitions_ready_to_wait_on_.at(pid) = true;
            cv_.notify_all();
        } else {
            RAPIDSMPF_EXPECTS(
                finished_chunk < chunk_goal, "finished chunk exceeds the goal"
            );
        }
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
        return partitions_ready_to_wait_on_.empty()
               || std::ranges::any_of(
                   partitions_ready_to_wait_on_,
                   [&](auto const& item) {
                       auto done = item.second;
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
    return extract_key(partitions_ready_to_wait_on_, finished_key);
}

void FinishCounter::wait_on(
    PartID pid, std::optional<std::chrono::milliseconds> timeout
) {
    std::unique_lock<std::mutex> lock(mutex_);
    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&]() {
        auto it = partitions_ready_to_wait_on_.find(pid);
        RAPIDSMPF_EXPECTS(
            it != partitions_ready_to_wait_on_.end(),
            "PartID has already been extracted",
            std::out_of_range
        );
        return it->second;
    });
    partitions_ready_to_wait_on_.erase(pid);
}

std::vector<PartID> FinishCounter::wait_some(
    std::optional<std::chrono::milliseconds> timeout
) {
    std::unique_lock<std::mutex> lock(mutex_);
    RAPIDSMPF_EXPECTS(
        !partitions_ready_to_wait_on_.empty(),
        "no more partitions to wait on",
        std::out_of_range
    );

    wait_for_if_timeout_else_wait(lock, cv_, timeout, [&]() {
        return std::ranges::any_of(partitions_ready_to_wait_on_, [](auto const& item) {
            return item.second;
        });
    });

    std::vector<PartID> result{};
    // TODO: hand-writing iteration rather than range-for to avoid
    // needing to rehash the key during extract_key. Needs
    // std::ranges, I think.
    for (auto it = partitions_ready_to_wait_on_.cbegin();
         it != partitions_ready_to_wait_on_.cend();)
    {
        // extract_key invalidates the iterator
        auto tmp = it++;
        if (tmp->second) {
            result.push_back(extract_key(partitions_ready_to_wait_on_, tmp));
        }
    }
    return result;
}


}  // namespace rapidsmpf::shuffler::detail
