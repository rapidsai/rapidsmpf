/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <vector>

#include <rapidsmpf/communicator/message_interface.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler {

/**
 * @brief Adapter that makes Chunk compatible with communicator::MessageInterface.
 *
 * This adapter allows existing Chunk-based code to work with the new communication
 * interface without requiring immediate changes to all Chunk usage.
 */
class ChunkMessageAdapter : public communicator::MessageInterface {
  public:
    /**
     * @brief Construct adapter from an existing Chunk.
     *
     * @param chunk The chunk to wrap.
     * @param peer_rank The destination for outgoing or source for incoming messages.
     */
    ChunkMessageAdapter(detail::Chunk chunk, Rank peer_rank);

    /**
     * @copydoc communicator::MessageInterface::message_id
     */
    [[nodiscard]] std::uint64_t message_id() const override;

    /**
     * @copydoc communicator::MessageInterface::peer_rank
     */
    [[nodiscard]] Rank peer_rank() const override;

    /**
     * @copydoc communicator::MessageInterface::serialize_metadata
     */
    [[nodiscard]] std::vector<std::uint8_t> serialize_metadata() const override;

    /**
     * @copydoc communicator::MessageInterface::total_data_size
     */
    [[nodiscard]] std::size_t total_data_size() const override;

    /**
     * @copydoc communicator::MessageInterface::is_data_ready
     */
    [[nodiscard]] bool is_data_ready() const override;

    /**
     * @copydoc communicator::MessageInterface::set_data_buffer
     */
    void set_data_buffer(std::unique_ptr<Buffer> buffer) override;

    /**
     * @copydoc communicator::MessageInterface::release_data_buffer
     */
    [[nodiscard]] std::unique_ptr<Buffer> release_data_buffer() override;

    /**
     * @copydoc communicator::MessageInterface::data_memory_type
     *
     * @throws std::runtime_error if the data buffer is not set.
     */
    [[nodiscard]] MemoryType data_memory_type() const override;

    /**
     * @copydoc communicator::MessageInterface::is_ready
     */
    [[nodiscard]] bool is_ready() const override;

    /**
     * @copydoc communicator::MessageInterface::to_string
     */
    [[nodiscard]] std::string to_string() const override;

    /**
     * @brief Release the underlying Chunk.
     *
     * @return The wrapped chunk with ownership transferred.
     */
    [[nodiscard]] detail::Chunk release_chunk() {
        return std::move(chunk_);
    }

  private:
    detail::Chunk chunk_;
    Rank peer_rank_{0};  // Will be set by communication interface
};

/**
 * @brief Factory for creating ChunkMessageAdapter instances.
 *
 * This factory allows the communication interface to create Chunk-based messages
 * without knowing about the Chunk type directly.
 */
class ChunkMessageFactory : public communicator::MessageFactory {
  public:
    /**
     * @brief Construct a new ChunkMessageFactory to transfer Chunk data.
     *
     * @param allocate_buffer_fn Function to allocate buffers for incoming data.
     */
    explicit ChunkMessageFactory(
        std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn
    );

    /**
     * @copydoc communicator::MessageFactory::create_from_metadata
     */
    [[nodiscard]] std::unique_ptr<communicator::MessageInterface> create_from_metadata(
        std::vector<std::uint8_t> const& metadata, Rank peer_rank
    ) const override;

    /**
     * @copydoc communicator::MessageFactory::allocate_receive_buffer
     */
    [[nodiscard]] std::unique_ptr<Buffer> allocate_receive_buffer(
        std::size_t size, communicator::MessageInterface const& message
    ) const override;

  private:
    std::function<std::unique_ptr<Buffer>(std::size_t)> allocate_buffer_fn_;
};

/**
 * @brief Helper function to convert Chunk vector to communicator::MessageInterface
 * vector.
 *
 * @param chunks Vector of chunks to convert.
 * @param peer_rank_fn Function to determine peer rank for each chunk.
 * @return Vector of communicator::MessageInterface instances wrapping the chunks.
 */
[[nodiscard]] std::vector<std::unique_ptr<communicator::MessageInterface>>
chunks_to_messages(
    std::vector<detail::Chunk>&& chunks,
    std::function<Rank(detail::Chunk const&)> peer_rank_fn
);

/**
 * @brief Helper function to convert communicator::MessageInterface vector back to Chunk
 * vector.
 *
 * @param messages Vector of communicator::MessageInterface instances (must be
 * ChunkMessageAdapter).
 * @return Vector of chunks extracted from the adapters.
 */
[[nodiscard]] std::vector<detail::Chunk> messages_to_chunks(
    std::vector<std::unique_ptr<communicator::MessageInterface>>&& messages
);

}  // namespace rapidsmpf::shuffler
