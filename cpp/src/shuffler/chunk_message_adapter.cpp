/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk_message_adapter.hpp>

namespace rapidsmpf::shuffler {

ChunkMessageAdapter::ChunkMessageAdapter(detail::Chunk chunk, Rank peer_rank)
    : chunk_(std::move(chunk)), peer_rank_(peer_rank) {}

std::uint64_t ChunkMessageAdapter::message_id() const {
    return static_cast<std::uint64_t>(chunk_.chunk_id());
}

Rank ChunkMessageAdapter::peer_rank() const {
    return peer_rank_;
}

std::vector<std::uint8_t> ChunkMessageAdapter::serialize_metadata() const {
    return *chunk_.serialize();
}

std::size_t ChunkMessageAdapter::total_data_size() const {
    return chunk_.concat_data_size();
}

bool ChunkMessageAdapter::is_data_ready() const {
    return chunk_.is_data_buffer_set();
}

void ChunkMessageAdapter::set_data_buffer(std::unique_ptr<Buffer> buffer) {
    chunk_.set_data_buffer(std::move(buffer));
}

std::unique_ptr<Buffer> ChunkMessageAdapter::release_data_buffer() {
    return chunk_.release_data_buffer();
}

MemoryType ChunkMessageAdapter::data_memory_type() const {
    RAPIDSMPF_EXPECTS(chunk_.is_data_buffer_set(), "Chunk data buffer is not set");
    return chunk_.data_memory_type();
}

bool ChunkMessageAdapter::is_ready() const {
    return chunk_.is_ready();
}

std::string ChunkMessageAdapter::to_string() const {
    return chunk_.str();
}

ChunkMessageFactory::ChunkMessageFactory(
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
)
    : allocate_buffer_fn_(std::move(allocate_buffer_fn)) {}

std::unique_ptr<communicator::MessageInterface> ChunkMessageFactory::create_from_metadata(
    std::vector<std::uint8_t> const& metadata, Rank peer_rank
) const {
    auto chunk = detail::Chunk::deserialize(metadata, false);
    return std::make_unique<ChunkMessageAdapter>(std::move(chunk), peer_rank);
}

std::unique_ptr<Buffer> ChunkMessageFactory::allocate_receive_buffer(
    std::size_t size, communicator::MessageInterface const& /* message */
) const {
    return allocate_buffer_fn_(size);
}

std::vector<std::unique_ptr<communicator::MessageInterface>> chunks_to_messages(
    std::vector<detail::Chunk>&& chunks,
    std::function<Rank(detail::Chunk const&)> peer_rank_fn
) {
    std::vector<std::unique_ptr<communicator::MessageInterface>> messages;
    messages.reserve(chunks.size());

    for (auto&& chunk : chunks) {
        auto peer_rank = peer_rank_fn(chunk);
        messages.push_back(
            std::make_unique<ChunkMessageAdapter>(std::move(chunk), peer_rank)
        );
    }

    return messages;
}

std::vector<detail::Chunk> messages_to_chunks(
    std::vector<std::unique_ptr<communicator::MessageInterface>>&& messages
) {
    std::vector<detail::Chunk> chunks;
    chunks.reserve(messages.size());

    for (auto&& message : messages) {
        auto* adapter = dynamic_cast<ChunkMessageAdapter*>(
            dynamic_cast<communicator::MessageInterface*>(message.get())
        );
        RAPIDSMPF_EXPECTS(adapter != nullptr, "Message is not a ChunkMessageAdapter");
        chunks.push_back(adapter->release_chunk());
    }

    return chunks;
}

}  // namespace rapidsmpf::shuffler
