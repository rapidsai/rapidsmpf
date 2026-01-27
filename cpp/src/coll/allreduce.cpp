/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::coll {

AllReduce::AllReduce(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    ReduceOperator reduce_operator,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    std::function<void(void)> finished_callback
)
    : reduce_operator_{std::move(reduce_operator)},
      br_{br},
      nranks_{comm->nranks()},
      gatherer_{
          std::move(comm),
          std::move(progress_thread),
          op_id,
          br,
          std::move(statistics),
          std::move(finished_callback)
      } {}

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
        std::cmp_equal(total, nranks_),
        "AllReduce expects exactly one contribution from each rank",
        std::runtime_error
    );

    // Determine target memory type based on operator type
    MemoryType target_mem_type =
        reduce_operator_.is_device() ? MemoryType::DEVICE : MemoryType::HOST;

    // Normalize all buffers to the target memory type
    for (auto& pd : gathered) {
        if (pd.data && pd.data->mem_type() != target_mem_type) {
            auto reservation = br_->reserve_or_fail(pd.data->size, target_mem_type);
            pd.data = br_->move(std::move(pd.data), reservation);
        }
    }

    // Start with rank 0's contribution as the left (only true with Ordered::YES).
    PackedData left = std::move(gathered[0]);

    for (std::size_t r = 1; std::cmp_less(r, nranks_); ++r) {
        reduce_operator_(left, std::move(gathered[r]));
    }

    return left;
}

}  // namespace rapidsmpf::coll
