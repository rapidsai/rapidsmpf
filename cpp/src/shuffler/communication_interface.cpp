/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <utility>

#include <rapidsmpf/shuffler/communication_interface.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler {

TagCommunicationInterface::TagCommunicationInterface(
    std::shared_ptr<Communicator> comm,
    OpID op_id,
    Rank rank,
    std::shared_ptr<Statistics> statistics
)
    : comm_(std::move(comm)),
      rank_(rank),
      ready_for_data_tag_{op_id, 1},
      metadata_tag_{op_id, 2},
      gpu_data_tag_{op_id, 3},
      statistics_{std::move(statistics)} {}

void TagCommunicationInterface::submit_outgoing_chunks(
    std::vector<detail::Chunk>&& chunks,
    std::function<Rank(PartID)> partition_owner,
    BufferResource* br
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    // Store chunks for sending and initiate metadata transmission
    for (auto&& chunk : chunks) {
        auto dst = partition_owner(chunk.part_id(0));
        log.trace("send metadata to ", dst, ": ", chunk);
        RAPIDSMPF_EXPECTS(dst != rank_, "sending chunk to ourselves");

        fire_and_forget_.push_back(
            comm_->send(chunk.serialize(), dst, metadata_tag_, br)
        );

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

    statistics_->add_duration_stat(
        "comms-interface-submit-outgoing-chunks", Clock::now() - t0
    );
}

std::vector<detail::Chunk> TagCommunicationInterface::process_communication(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    auto const t0 = Clock::now();

    // Process all phases of the communication protocol
    receive_metadata_phase();
    setup_data_receives_phase(allocate_buffer_fn, stream, br);
    process_ready_acks_phase();
    auto completed_chunks = complete_data_transfers_phase();
    cleanup_completed_operations();

    statistics_->add_duration_stat(
        "comms-interface-process-communication-total", Clock::now() - t0
    );

    return completed_chunks;
}

bool TagCommunicationInterface::is_idle() const {
    return fire_and_forget_.empty() && incoming_chunks_.empty()
           && outgoing_chunks_.empty() && in_transit_chunks_.empty()
           && in_transit_futures_.empty()
           && std::all_of(
               ready_ack_receives_.begin(),
               ready_ack_receives_.end(),
               [](const auto& kv) { return kv.second.empty(); }
           );
}

void TagCommunicationInterface::receive_metadata_phase() {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    while (true) {
        auto const [msg, src] = comm_->recv_any(metadata_tag_);
        if (!msg)
            break;

        auto chunk = detail::Chunk::deserialize(*msg, false);
        log.trace("recv_any from ", src, ": ", chunk);
        incoming_chunks_.insert({src, std::move(chunk)});
    }

    statistics_->add_duration_stat("comms-interface-receive-metadata", Clock::now() - t0);
}

void TagCommunicationInterface::setup_data_receives_phase(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn,
    rmm::cuda_stream_view /* stream */,
    BufferResource* br
) {
    auto& log = comm_->logger();
    auto const t0 = Clock::now();

    for (auto it = incoming_chunks_.begin(); it != incoming_chunks_.end();) {
        auto& [src, chunk] = *it;
        log.trace("checking incoming chunk data from ", src, ": ", chunk);

        if (chunk.concat_data_size() > 0) {
            if (!chunk.is_data_buffer_set()) {
                chunk.set_data_buffer(allocate_buffer_fn(chunk.concat_data_size()));
            }

            if (!chunk.is_ready()) {
                ++it;
                continue;
            }

            // Extract the chunk and set up for data transfer
            auto [src, chunk] = extract_item(incoming_chunks_, it++);

            auto future = comm_->recv(src, gpu_data_tag_, chunk.release_data_buffer());
            RAPIDSMPF_EXPECTS(
                in_transit_futures_.insert({chunk.chunk_id(), std::move(future)}).second,
                "in transit future already exists"
            );
            RAPIDSMPF_EXPECTS(
                in_transit_chunks_.insert({chunk.chunk_id(), std::move(chunk)}).second,
                "in transit chunk already exists"
            );

            fire_and_forget_.push_back(comm_->send(
                detail::ReadyForDataMessage{chunk.chunk_id()}.pack(),
                src,
                ready_for_data_tag_,
                br
            ));
        } else {
            // Control/metadata-only chunk - will be handled in
            // complete_data_transfers_phase
            ++it;
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-setup-data-receives", Clock::now() - t0
    );
}

void TagCommunicationInterface::process_ready_acks_phase() {
    auto const t0 = Clock::now();

    for (auto& [dst, futures] : ready_ack_receives_) {
        auto finished = comm_->test_some(futures);
        for (auto&& future : finished) {
            auto const msg_data = comm_->get_gpu_data(std::move(future));
            auto msg = detail::ReadyForDataMessage::unpack(
                const_cast<Buffer const&>(*msg_data).host()
            );
            auto chunk = extract_value(outgoing_chunks_, msg.cid);

            fire_and_forget_.push_back(
                comm_->send(chunk.release_data_buffer(), dst, gpu_data_tag_)
            );
        }
    }

    statistics_->add_duration_stat(
        "comms-interface-process-ready-acks", Clock::now() - t0
    );
}

std::vector<detail::Chunk> TagCommunicationInterface::complete_data_transfers_phase() {
    auto const t0 = Clock::now();

    std::vector<detail::Chunk> completed_chunks;

    // Handle completed data transfers - use the same approach as the original code
    if (!in_transit_futures_.empty()) {
        std::vector<rapidsmpf::shuffler::detail::ChunkID> finished =
            comm_->test_some(in_transit_futures_);
        for (auto cid : finished) {
            auto chunk = extract_value(in_transit_chunks_, cid);
            auto future = extract_value(in_transit_futures_, cid);
            chunk.set_data_buffer(comm_->get_gpu_data(std::move(future)));
            completed_chunks.push_back(std::move(chunk));
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

    statistics_->add_duration_stat(
        "comms-interface-complete-data-transfers", Clock::now() - t0
    );

    return completed_chunks;
}

void TagCommunicationInterface::cleanup_completed_operations() {
    if (!fire_and_forget_.empty()) {
        std::ignore = comm_->test_some(fire_and_forget_);
    }
}

}  // namespace rapidsmpf::shuffler
