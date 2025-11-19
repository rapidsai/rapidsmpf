/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <rapidsmpf/allreduce/allreduce.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::allreduce {

AllReduce::AllReduce(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    ReduceKernel reduce_kernel,
    std::function<void(void)> finished_callback
)
    : comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      br_{br},
      statistics_{std::move(statistics)},
      reduce_kernel_{std::move(reduce_kernel)},
      finished_callback_{std::move(finished_callback)},
      gatherer_{comm_, progress_thread_, op_id, br_, statistics_} {
    RAPIDSMPF_EXPECTS(
        static_cast<bool>(reduce_kernel_),
        "AllReduce requires a valid ReduceKernel at construction time"
    );
}

AllReduce::~AllReduce() = default;

void AllReduce::insert(std::uint64_t sequence_number, PackedData&& packed_data) {
    nlocal_insertions_.fetch_add(1, std::memory_order_relaxed);
    gatherer_.insert(sequence_number, std::move(packed_data));
}

void AllReduce::insert_finished() {
    gatherer_.insert_finished();
}

bool AllReduce::finished() const noexcept {
    return gatherer_.finished();
}

std::vector<PackedData> AllReduce::wait_and_extract(std::chrono::milliseconds timeout) {
    // Block until the underlying allgather completes, then perform the reduction locally
    // (exactly once).
    if (!reduced_computed_.load(std::memory_order_acquire)) {
        auto gathered = gatherer_.wait_and_extract(
            allgather::AllGather::Ordered::YES, std::move(timeout)
        );
        reduced_results_ = reduce_all(std::move(gathered));
        reduced_computed_.store(true, std::memory_order_release);
        if (finished_callback_) {
            finished_callback_();
        }
    }
    return std::move(reduced_results_);
}

bool AllReduce::is_ready() const noexcept {
    return reduced_computed_.load(std::memory_order_acquire) || gatherer_.finished();
}

std::vector<PackedData> AllReduce::reduce_all(std::vector<PackedData>&& gathered) {
    auto const nranks = static_cast<std::size_t>(comm_->nranks());
    auto const total = gathered.size();

    if (total == 0) {
        return {};
    }

    RAPIDSMPF_EXPECTS(
        nranks > 0, "AllReduce requires a positive number of ranks", std::runtime_error
    );
    RAPIDSMPF_EXPECTS(
        total % nranks == 0,
        "AllReduce expects each rank to contribute the same number of messages",
        std::runtime_error
    );

    auto const n_local =
        static_cast<std::size_t>(nlocal_insertions_.load(std::memory_order_acquire));
    auto const n_per_rank = total / nranks;

    // We allow non-uniform insertion counts across ranks but require that the local
    // insertion count matches the per-rank contribution implied by the gather.
    RAPIDSMPF_EXPECTS(
        n_local == 0 || n_local == n_per_rank,
        "AllReduce local insertion count does not match gathered contributions per rank",
        std::runtime_error
    );

    std::vector<PackedData> results;
    results.reserve(n_per_rank);

    // Conceptually, the k-th insertion on each rank participates in a single
    // reduction. With ordered allgather results, entries are laid out as:
    //   [rank0:0..n_per_rank-1][rank1:0..n_per_rank-1]...[rankP-1:0..n_per_rank-1]
    for (std::size_t k = 0; k < n_per_rank; ++k) {
        // Start from rank 0's contribution for this logical insertion.
        auto accum = std::move(gathered[k]);
        for (std::size_t r = 1; r < nranks; ++r) {
            auto idx = r * n_per_rank + k;
            reduce_kernel_(accum, std::move(gathered[idx]));
        }
        results.emplace_back(std::move(accum));
    }

    return results;
}

}  // namespace rapidsmpf::allreduce
