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

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::coll {

AllReduce::AllReduce(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    ReduceKernel reduce_kernel,
    std::function<void(void)> finished_callback
)
    : reduce_kernel_{std::move(reduce_kernel)},
      finished_callback_{std::move(finished_callback)},
      nranks_{comm->nranks()},
      gatherer_{
          std::move(comm), std::move(progress_thread), op_id, br, std::move(statistics)
      } {
    RAPIDSMPF_EXPECTS(
        static_cast<bool>(reduce_kernel_),
        "AllReduce requires a valid ReduceKernel at construction time"
    );
}

AllReduce::~AllReduce() = default;

void AllReduce::insert(PackedData&& packed_data) {
    RAPIDSMPF_EXPECTS(
        !inserted_,
        "AllReduce::insert can only be called once per instance",
        std::runtime_error
    );
    inserted_ = true;
    gatherer_.insert(0, std::move(packed_data));
    gatherer_.insert_finished();
}

bool AllReduce::finished() const noexcept {
    return gatherer_.finished();
}

PackedData AllReduce::wait_and_extract(std::chrono::milliseconds timeout) {
    // Block until the underlying allgather completes, then perform the reduction locally
    // (exactly once).
    auto gathered =
        gatherer_.wait_and_extract(AllGather::Ordered::YES, std::move(timeout));
    return reduce_all(std::move(gathered));
}

bool AllReduce::is_ready() const noexcept {
    return gatherer_.finished();
}

PackedData AllReduce::reduce_all(std::vector<PackedData>&& gathered) {
    auto const total = gathered.size();

    RAPIDSMPF_EXPECTS(
        total == static_cast<std::size_t>(nranks_),
        "AllReduce expects exactly one contribution from each rank",
        std::runtime_error
    );

    // Start with rank 0's contribution as the accumulator
    auto accum = std::move(gathered[0]);

    // Reduce contributions from all other ranks into the accumulator
    for (std::size_t r = 1; r < static_cast<std::size_t>(nranks_); ++r) {
        reduce_kernel_(accum, std::move(gathered[r]));
    }

    return accum;
}

}  // namespace rapidsmpf::coll
