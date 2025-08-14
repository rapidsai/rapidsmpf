/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <utility>

#include <rapidsmpf/shuffler/communication_interface.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler {

DefaultShufflerCommunication::DefaultShufflerCommunication(
    std::shared_ptr<Communicator> comm, OpID op_id, Rank rank
)
    : comm_(std::move(comm)),
      rank_(rank),
      ready_for_data_tag_{op_id, 1},
      metadata_tag_{op_id, 2},
      gpu_data_tag_{op_id, 3} {
    statistics_["metadata_sent"] = 0;
    statistics_["metadata_received"] = 0;
    statistics_["data_sent"] = 0;
    statistics_["data_received"] = 0;
    statistics_["ready_signals_sent"] = 0;
    statistics_["ready_signals_received"] = 0;
}

void DefaultShufflerCommunication::submit_outgoing_chunks(
    std::vector<detail::Chunk>&& chunks,
    std::function<Rank(PartID)> partition_owner,
    BufferResource* br
) {
    // Store chunks for sending and initiate metadata transmission
    for (auto&& chunk : chunks) {
        auto dst = partition_owner(chunk.part_id(0));
        RAPIDSMPF_EXPECTS(dst != rank_, "sending chunk to ourselves");

        fire_and_forget_.push_back(
            comm_->send(chunk.serialize(), dst, metadata_tag_, br)
        );
        statistics_["metadata_sent"]++;

        if (chunk.concat_data_size() > 0) {
            RAPIDSMPF_EXPECTS(
                outgoing_chunks_.insert({chunk.chunk_id(), std::move(chunk)}).second,
                "outgoing chunk already exists"
            );
            ready_ack_receives_[dst].push_back(comm_->recv(
                dst,
                ready_for_data_tag_,
                br->move(
                    std::make_unique<std::vector<std::uint8_t>>(
                        detail::ReadyForDataMessage::byte_size
                    )
                )
            ));
        }
    }
}

std::vector<detail::Chunk> DefaultShufflerCommunication::process_communication(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    // Process all phases of the communication protocol
    receive_metadata_phase();
    setup_data_receives_phase(allocate_buffer_fn, stream, br);
    process_ready_acks_phase();
    auto completed_chunks = complete_data_transfers_phase();
    cleanup_completed_operations();

    return completed_chunks;
}

bool DefaultShufflerCommunication::is_idle() const {
    return fire_and_forget_.empty() && incoming_chunks_.empty()
           && outgoing_chunks_.empty() && in_transit_chunks_.empty()
           && in_transit_futures_.empty()
           && std::all_of(
               ready_ack_receives_.begin(),
               ready_ack_receives_.end(),
               [](const auto& kv) { return kv.second.empty(); }
           );
}

std::unordered_map<std::string, std::size_t>
DefaultShufflerCommunication::get_statistics() const {
    return statistics_;
}

void DefaultShufflerCommunication::receive_metadata_phase() {
    while (true) {
        auto const [msg, src] = comm_->recv_any(metadata_tag_);
        if (!msg)
            break;

        auto chunk = detail::Chunk::deserialize(*msg, false);
        statistics_["metadata_received"]++;
        incoming_chunks_.insert({src, std::move(chunk)});
    }
}

void DefaultShufflerCommunication::setup_data_receives_phase(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
    rmm::cuda_stream_view /* stream */,
    BufferResource* br
) {
    for (auto it = incoming_chunks_.begin(); it != incoming_chunks_.end();) {
        auto& [src, chunk] = *it;

        if (chunk.concat_data_size() > 0) {
            if (!chunk.is_data_buffer_set()) {
                chunk.set_data_buffer(allocate_buffer_fn(chunk.concat_data_size()));
            }

            if (!chunk.is_ready()) {
                ++it;
                continue;
            }

            // Extract the chunk and set up for data transfer
            auto src_rank = src;
            auto chunk_id = chunk.chunk_id();
            auto data_buffer = chunk.release_data_buffer();
            auto [extracted_src, extracted_chunk] = extract_item(incoming_chunks_, it++);

            auto future = comm_->recv(src_rank, gpu_data_tag_, std::move(data_buffer));
            RAPIDSMPF_EXPECTS(
                in_transit_futures_.insert({chunk_id, std::move(future)}).second,
                "in transit future already exists"
            );
            RAPIDSMPF_EXPECTS(
                in_transit_chunks_.insert({chunk_id, std::move(extracted_chunk)}).second,
                "in transit chunk already exists"
            );

            fire_and_forget_.push_back(comm_->send(
                detail::ReadyForDataMessage{chunk_id}.pack(),
                src_rank,
                ready_for_data_tag_,
                br
            ));
            statistics_["ready_signals_sent"]++;
        } else {
            // Control/metadata-only chunk - will be handled in
            // complete_data_transfers_phase
            ++it;
        }
    }
}

void DefaultShufflerCommunication::process_ready_acks_phase() {
    for (auto& [dst, futures] : ready_ack_receives_) {
        auto finished = comm_->test_some(futures);
        for (auto&& future : finished) {
            auto const msg_data = comm_->get_gpu_data(std::move(future));
            auto msg = detail::ReadyForDataMessage::unpack(
                const_cast<Buffer const&>(*msg_data).host()
            );
            auto chunk = extract_value(outgoing_chunks_, msg.cid);
            statistics_["ready_signals_received"]++;
            statistics_["data_sent"]++;

            fire_and_forget_.push_back(
                comm_->send(chunk.release_data_buffer(), dst, gpu_data_tag_)
            );
        }
    }
}

std::vector<detail::Chunk> DefaultShufflerCommunication::complete_data_transfers_phase() {
    std::vector<detail::Chunk> completed_chunks;

    // Handle completed data transfers - use the same approach as the original code
    if (!in_transit_futures_.empty()) {
        // Convert futures to vector and track chunk IDs
        std::vector<std::unique_ptr<Communicator::Future>> futures_vec;
        std::vector<detail::ChunkID> chunk_ids_vec;

        // Extract futures and store chunk IDs
        for (auto it = in_transit_futures_.begin(); it != in_transit_futures_.end(); ++it)
        {
            futures_vec.push_back(std::move(it->second));
            chunk_ids_vec.push_back(it->first);
        }
        in_transit_futures_.clear();

        // Test for completed futures
        auto completed_futures = comm_->test_some(futures_vec);

        // Process completed futures
        // Since we have parallel arrays, completed futures are from the end
        size_t num_completed = completed_futures.size();
        size_t original_size = chunk_ids_vec.size();

        for (size_t i = 0; i < num_completed; ++i) {
            // Get the chunk ID from the end of the array (test_some processes from end)
            auto chunk_id = chunk_ids_vec[original_size - num_completed + i];

            auto chunk = extract_value(in_transit_chunks_, chunk_id);
            chunk.set_data_buffer(comm_->get_gpu_data(std::move(completed_futures[i])));
            statistics_["data_received"]++;
            completed_chunks.push_back(std::move(chunk));
        }

        // Put back any remaining futures (those that weren't completed)
        for (size_t i = 0; i < futures_vec.size(); ++i) {
            auto chunk_id = chunk_ids_vec[i];
            in_transit_futures_[chunk_id] = std::move(futures_vec[i]);
        }
    }

    // Handle control/metadata-only chunks from incoming_chunks_
    for (auto it = incoming_chunks_.begin(); it != incoming_chunks_.end();) {
        auto& [src, chunk] = *it;
        if (chunk.concat_data_size() == 0) {
            auto [src, chunk] = extract_item(incoming_chunks_, it++);
            completed_chunks.push_back(std::move(chunk));
        } else {
            ++it;
        }
    }

    return completed_chunks;
}

void DefaultShufflerCommunication::cleanup_completed_operations() {
    if (!fire_and_forget_.empty()) {
        std::ignore = comm_->test_some(fire_and_forget_);
    }
}

}  // namespace rapidsmpf::shuffler
