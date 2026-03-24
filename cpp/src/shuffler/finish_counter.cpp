/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>

namespace rapidsmpf::shuffler::detail {

FinishCounter::FinishCounter(Rank nranks, PartID n_local_partitions)
    : nranks_{n_local_partitions > 0 ? nranks : 0},
      rank_reported_(safe_cast<std::size_t>(nranks_), false) {}

bool FinishCounter::all_finished() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return n_ranks_with_goalpost_ == nranks_
           && total_finished_chunks_ == total_chunk_goal_;
}

void FinishCounter::move_goalpost(Rank src_rank, ChunkID nchunks) {
    std::unique_lock<std::mutex> lock(mutex_);
    RAPIDSMPF_EXPECTS(nchunks != 0, "the goalpost was moved by 0 chunks");
    RAPIDSMPF_EXPECTS(src_rank < nranks_, "Invalid source rank in move_goalpost");
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
}

std::string detail::FinishCounter::str() const {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "FinishCounter(n_ranks_with_goalpost=" << n_ranks_with_goalpost_
       << ", total_chunk_goal=" << total_chunk_goal_
       << ", total_finished_chunks=" << total_finished_chunks_ << ")";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
