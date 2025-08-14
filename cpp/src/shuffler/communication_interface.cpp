/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/shuffler/communication_interface.hpp>

namespace rapidsmpf::shuffler {

DefaultShufflerCommunication::DefaultShufflerCommunication(
    std::shared_ptr<Communicator> comm, OpID op_id
)
    : comm_(std::move(comm)),
      ready_for_data_tag_{op_id, 1},
      metadata_tag_{op_id, 2},
      gpu_data_tag_{op_id, 3} {}

std::unique_ptr<Communicator::Future> DefaultShufflerCommunication::send_chunk_metadata(
    std::unique_ptr<std::vector<uint8_t>> serialized_metadata,
    Rank dest_rank,
    BufferResource* br
) {
    return comm_->send(std::move(serialized_metadata), dest_rank, metadata_tag_, br);
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank>
DefaultShufflerCommunication::receive_chunk_metadata() {
    return comm_->recv_any(metadata_tag_);
}

std::unique_ptr<Communicator::Future> DefaultShufflerCommunication::send_ready_for_data(
    std::unique_ptr<std::vector<uint8_t>> ready_msg, Rank dest_rank, BufferResource* br
) {
    return comm_->send(std::move(ready_msg), dest_rank, ready_for_data_tag_, br);
}

std::unique_ptr<Communicator::Future>
DefaultShufflerCommunication::receive_ready_for_data(
    Rank source_rank, std::unique_ptr<Buffer> buffer
) {
    return comm_->recv(source_rank, ready_for_data_tag_, std::move(buffer));
}

std::unique_ptr<Communicator::Future> DefaultShufflerCommunication::send_gpu_data(
    std::unique_ptr<Buffer> data_buffer, Rank dest_rank
) {
    return comm_->send(std::move(data_buffer), dest_rank, gpu_data_tag_);
}

std::unique_ptr<Communicator::Future> DefaultShufflerCommunication::receive_gpu_data(
    Rank source_rank, std::unique_ptr<Buffer> data_buffer
) {
    return comm_->recv(source_rank, gpu_data_tag_, std::move(data_buffer));
}

std::vector<std::unique_ptr<Communicator::Future>>
DefaultShufflerCommunication::test_some(
    std::vector<std::unique_ptr<Communicator::Future>>& futures
) {
    return comm_->test_some(futures);
}

std::unique_ptr<Buffer> DefaultShufflerCommunication::get_gpu_data(
    std::unique_ptr<Communicator::Future> future
) {
    return comm_->get_gpu_data(std::move(future));
}

}  // namespace rapidsmpf::shuffler
