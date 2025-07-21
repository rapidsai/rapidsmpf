/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/all_gatherer/all_gatherer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::experimental::all_gatherer {

/**
 * @brief A naive implementation of the all-gather operation.
 *
 * The implementation creates a copy of the data for each peer during insertion and assign
 * the peer's rank as the partition ID, ie. there will only be one partition per rank.
 * `partition_owner` lambda function is an identity function. These enables the data
 * copies to be sent to the corresponding rank using the Shuffler.
 *
 * @note This implementation might not be scalable, as it makes a copy of the data for
 * each peer during insertion.
 */

AllGatherer::AllGatherer(
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

AllGatherer::~AllGatherer() {
    shutdown();
}

void AllGatherer::shutdown() {
    if (shuffler_) {
        shuffler_->shutdown();
        shuffler_.reset();
    }
}

void AllGatherer::insert(PackedData&& data) {
    std::unordered_map<shuffler::PartID, PackedData> chunks;
    chunks.reserve(static_cast<size_t>(comm_->nranks()));

    for (Rank r = 0; r < comm_->nranks(); ++r) {
        if (r == comm_->rank()) {
            continue;
        }
        // copy data in the chunk
        auto metadata_buf = std::make_unique<std::vector<uint8_t>>(*data.metadata);
        auto data_buf = std::make_unique<rmm::device_buffer>(
            data.gpu_data->data(), data.gpu_data->size(), stream_, br_->device_mr()
        );
        chunks.emplace(
            static_cast<shuffler::PartID>(r),
            PackedData{std::move(metadata_buf), std::move(data_buf)}
        );
    }
    chunks.emplace(comm_->rank(), std::move(data));

    shuffler_->insert(std::move(chunks));
}

void AllGatherer::insert_finished() {
    // there will be n_ranks partitions
    std::vector<shuffler::PartID> pids(static_cast<size_t>(comm_->nranks()));
    std::iota(pids.begin(), pids.end(), 0);
    shuffler_->insert_finished(std::move(pids));
}

bool AllGatherer::finished() const {
    return shuffler_->finished();
}

std::vector<PackedData> AllGatherer::wait_and_extract(
    std::optional<std::chrono::milliseconds> timeout
) {
    // wait for the local partition data
    auto pid = static_cast<shuffler::PartID>(comm_->rank());
    shuffler_->wait_on(pid, timeout);
    return shuffler_->extract(pid);
}

}  // namespace rapidsmpf::experimental::all_gatherer
