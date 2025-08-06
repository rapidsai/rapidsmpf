/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/all_gather/all_gather.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::all_gather {

/**
 * @brief A naive implementation of the all-gather operation.
 *
 * The implementation creates a copy of the data for each peer during insertion and assign
 * the peer's rank as the partition ID, i.e., there will only be one partition per rank.
 * `partition_owner` lambda function is an identity function. These enable the data
 * copies to be sent to the corresponding rank using the Shuffler.
 *
 * @note This implementation might not be scalable, as it makes a copy of the data for
 * each peer during insertion.
 */

AllGather::AllGather(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
)
    : comm_(comm.get()),
      shuffler_(std::make_unique<shuffler::Shuffler>(
          std::move(comm),
          std::move(progress_thread),
          op_id,
          comm_->nranks(),
          stream,
          br,
          std::move(statistics),
          [](std::shared_ptr<Communicator> const&, shuffler::PartID pid) -> Rank {
              // identity-like mapping, as there are only n_ranks partitions
              return static_cast<Rank>(pid);
          }
      )),
      stream_(stream),
      br_(br) {}

AllGather::~AllGather() {
    shutdown();
}

void AllGather::shutdown() {
    if (shuffler_) {
        shuffler_->shutdown();
        shuffler_.reset();
    }
}

void AllGather::insert(PackedData&& data) {
    std::unordered_map<shuffler::PartID, PackedData> chunks;
    chunks.reserve(static_cast<size_t>(comm_->nranks()));

    for (Rank r = 0; r < comm_->nranks(); ++r) {
        if (r == comm_->rank()) {
            continue;
        }
        // copy data in the chunk
        auto metadata_buf = std::make_unique<std::vector<uint8_t>>(*data.metadata);
        auto res = reserve_or_fail(br_, data.data->size);
        auto data_buf = br_->copy(data.data, stream_, res);
        chunks.emplace(
            static_cast<shuffler::PartID>(r),
            PackedData{std::move(metadata_buf), std::move(data_buf)}
        );
    }
    chunks.emplace(static_cast<shuffler::PartID>(comm_->rank()), std::move(data));

    shuffler_->insert(std::move(chunks));
}

void AllGather::insert_finished() {
    bool expected = false;
    if (insert_finished_.compare_exchange_strong(expected, true)) {
        // there will be n_ranks partitions
        std::vector<shuffler::PartID> pids(static_cast<size_t>(comm_->nranks()));
        std::iota(pids.begin(), pids.end(), 0);
        shuffler_->insert_finished(std::move(pids));
    }
}

bool AllGather::finished() const {
    return shuffler_->finished();
}

std::vector<PackedData> AllGather::wait_and_extract(
    std::optional<std::chrono::milliseconds> timeout
) {
    RAPIDSMPF_EXPECTS(
        insert_finished_.load(), "insertion has not finished yet.", std::runtime_error
    );

    // wait for the local partition data
    auto pid = static_cast<shuffler::PartID>(comm_->rank());
    shuffler_->wait_on(pid, timeout);
    return shuffler_->extract(pid);
}

std::pair<std::vector<PackedData>, std::vector<uint64_t>>
AllGather::wait_and_extract_ordered(std::optional<std::chrono::milliseconds> timeout) {
    RAPIDSMPF_EXPECTS(
        insert_finished_.load(), "insertion has not finished yet.", std::runtime_error
    );

    // wait for the local partition data
    auto pid = static_cast<shuffler::PartID>(comm_->rank());
    shuffler_->wait_on(pid, std::move(timeout));

    auto chunks = shuffler_->extract_chunks(pid);
    std::ranges::sort(chunks, [](const auto& lhs, const auto& rhs) {
        return lhs.chunk_id() < rhs.chunk_id();
    });

    std::vector<uint64_t> n_chunks_per_rank(static_cast<size_t>(comm_->nranks()), 0);
    std::vector<PackedData> packed_data;
    packed_data.reserve(chunks.size());

    for (auto&& chunk : chunks) {
        auto rank = shuffler::Shuffler::extract_rank(chunk.chunk_id());
        n_chunks_per_rank[static_cast<size_t>(rank)]++;
        packed_data.emplace_back(
            std::move(chunk.release_metadata_buffer()),
            std::move(chunk.release_data_buffer())
        );
    }

    return {std::move(packed_data), std::move(n_chunks_per_rank)};
}

}  // namespace rapidsmpf::all_gather
