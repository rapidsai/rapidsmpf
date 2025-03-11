/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/finish_counter.hpp>

namespace rapidsmp::shuffler::detail {

FinishCounter::FinishCounter(Rank nranks, std::vector<PartID> const& local_partitions)
    : nranks_{nranks} {
    // Initially, none of the partitions are ready to wait on.
    for (auto pid : local_partitions) {
        partitions_ready_to_wait_on_.insert({pid, false});
    }
}

void FinishCounter::move_goalpost(Rank rank, PartID pid, ChunkID nchunks) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto& [rank_counter, chunk_goal] = goalposts_[pid];
    RAPIDSMP_EXPECTS(
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
            RAPIDSMP_EXPECTS(
                finished_chunk < chunk_goal, "finished chunk exceeds the goal"
            );
        }
    }
}

PartID FinishCounter::wait_any() {
    std::unique_lock<std::mutex> lock(mutex_);
    PartID finished_key{std::numeric_limits<PartID>::max()};

    cv_.wait(lock, [&]() {
        return partitions_ready_to_wait_on_.empty()
               || std::any_of(
                   partitions_ready_to_wait_on_.cbegin(),
                   partitions_ready_to_wait_on_.cend(),
                   [&](auto const& item) {
                       auto done = item.second;
                       if (done) {
                           finished_key = item.first;
                       }
                       return done;
                   }
               );
    });

    RAPIDSMP_EXPECTS(
        finished_key != std::numeric_limits<PartID>::max(),
        "no more partitions to wait on",
        std::out_of_range
    );

    // We extract the partition to avoid returning the same partition twice.
    return extract_key(partitions_ready_to_wait_on_, finished_key);
}

void FinishCounter::wait_on(PartID pid) {
    auto predicate = [&]() {
        auto it = partitions_ready_to_wait_on_.find(pid);
        RAPIDSMP_EXPECTS(
            it != partitions_ready_to_wait_on_.end(),
            "PartID has already been extracted",
            std::out_of_range
        );
        return it->second;
    };
    std::unique_lock<std::mutex> lock(mutex_);
    if (!predicate()) {
        cv_.wait(lock, predicate);
    }
    partitions_ready_to_wait_on_.erase(pid);
}

std::vector<PartID> FinishCounter::wait_some() {
    std::unique_lock<std::mutex> lock(mutex_);
    RAPIDSMP_EXPECTS(
        !partitions_ready_to_wait_on_.empty(),
        "no more partitions to wait on",
        std::out_of_range
    );
    cv_.wait(lock, [&]() {
        return std::any_of(
            partitions_ready_to_wait_on_.begin(),
            partitions_ready_to_wait_on_.end(),
            [](auto const& item) { return item.second; }
        );
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


}  // namespace rapidsmp::shuffler::detail
