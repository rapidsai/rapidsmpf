/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler {

/**
 * @brief Abstract interface for shuffler communication operations.
 *
 * This interface abstracts the communication patterns used in the shuffler's main
 * progress loop. It provides a clean separation between the shuffler logic and the
 * underlying communication implementation, allowing for different communication
 * strategies while maintaining the same high-level shuffler behavior.
 *
 * The communication pattern involves:
 * 1. Sending serialized chunk metadata to notify receivers about incoming data
 * 2. Receiving serialized chunk metadata from other ranks
 * 3. Sending "ready for data" acknowledgments to indicate readiness to receive GPU data
 * 4. Receiving "ready for data" acknowledgments from receivers
 * 5. Sending GPU data buffers to receivers
 * 6. Receiving GPU data buffers from senders
 */
class ShufflerCommunicationInterface {
  public:
    virtual ~ShufflerCommunicationInterface() = default;

    /**
     * @brief Send serialized chunk metadata to a destination rank.
     *
     * This is step 1 of the communication protocol - notifying the receiver about an
     * incoming chunk and its metadata.
     *
     * @param serialized_metadata The serialized chunk metadata buffer.
     * @param dest_rank The destination rank to send to.
     * @param br Buffer resource for managing the send operation.
     * @return A future representing the ongoing send operation.
     */
    virtual std::unique_ptr<Communicator::Future> send_chunk_metadata(
        std::unique_ptr<std::vector<uint8_t>> serialized_metadata,
        Rank dest_rank,
        BufferResource* br
    ) = 0;

    /**
     * @brief Receive any available serialized chunk metadata.
     *
     * This is step 2 of the communication protocol - receiving chunk metadata from
     * any sender rank.
     *
     * @return A pair containing the received metadata buffer and the source rank.
     *         Returns {nullptr, invalid_rank} if no message is available.
     */
    virtual std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank>
    receive_chunk_metadata() = 0;

    /**
     * @brief Send a "ready for data" acknowledgment to a source rank.
     *
     * This is step 3 of the communication protocol - informing the sender that we
     * are ready to receive the GPU data for a specific chunk.
     *
     * @param ready_msg The ready-for-data message containing the chunk ID.
     * @param dest_rank The destination rank to send the acknowledgment to.
     * @param br Buffer resource for managing the send operation.
     * @return A future representing the ongoing send operation.
     */
    virtual std::unique_ptr<Communicator::Future> send_ready_for_data(
        std::unique_ptr<std::vector<uint8_t>> ready_msg,
        Rank dest_rank,
        BufferResource* br
    ) = 0;

    /**
     * @brief Post a receive operation for a "ready for data" acknowledgment.
     *
     * This is step 4 of the communication protocol - setting up to receive
     * acknowledgments from receivers indicating they are ready for GPU data.
     *
     * @param source_rank The source rank to receive from.
     * @param buffer Buffer to receive the acknowledgment into.
     * @return A future representing the ongoing receive operation.
     */
    virtual std::unique_ptr<Communicator::Future> receive_ready_for_data(
        Rank source_rank, std::unique_ptr<Buffer> buffer
    ) = 0;

    /**
     * @brief Send GPU data buffer to a destination rank.
     *
     * This is step 5 of the communication protocol - sending the actual GPU data
     * after receiving a ready-for-data acknowledgment.
     *
     * @param data_buffer The GPU data buffer to send.
     * @param dest_rank The destination rank to send to.
     * @return A future representing the ongoing send operation.
     */
    virtual std::unique_ptr<Communicator::Future> send_gpu_data(
        std::unique_ptr<Buffer> data_buffer, Rank dest_rank
    ) = 0;

    /**
     * @brief Post a receive operation for GPU data from a source rank.
     *
     * This is step 6 of the communication protocol - receiving the actual GPU data
     * after sending a ready-for-data acknowledgment.
     *
     * @param source_rank The source rank to receive from.
     * @param data_buffer Buffer to receive the GPU data into.
     * @return A future representing the ongoing receive operation.
     */
    virtual std::unique_ptr<Communicator::Future> receive_gpu_data(
        Rank source_rank, std::unique_ptr<Buffer> data_buffer
    ) = 0;

    /**
     * @brief Test completion of multiple futures and return completed ones.
     *
     * @param futures Container of futures to test, completed futures are removed.
     * @return Vector of completed futures.
     */
    virtual std::vector<std::unique_ptr<Communicator::Future>> test_some(
        std::vector<std::unique_ptr<Communicator::Future>>& futures
    ) = 0;

    /**
     * @brief Test completion of futures mapped by a key type.
     *
     * @tparam KeyType The type of the key used in the map.
     * @param futures_map Map of futures to test, completed futures are removed.
     * @return Vector of keys whose futures completed.
     */
    template <typename KeyType>
    std::vector<KeyType> test_some(
        std::unordered_map<KeyType, std::unique_ptr<Communicator::Future>>& futures_map
    ) {
        std::vector<KeyType> completed_keys;
        for (auto it = futures_map.begin(); it != futures_map.end();) {
            if (it->second->is_ready()) {
                completed_keys.push_back(it->first);
                it = futures_map.erase(it);
            } else {
                ++it;
            }
        }
        return completed_keys;
    }

    /**
     * @brief Extract the GPU data buffer from a completed future.
     *
     * @param future The completed future to extract data from.
     * @return The GPU data buffer.
     */
    virtual std::unique_ptr<Buffer> get_gpu_data(
        std::unique_ptr<Communicator::Future> future
    ) = 0;
};

/**
 * @brief Default implementation of ShufflerCommunicationInterface.
 *
 * This implementation uses the existing communicator API and replicates the current
 * communication behavior of the shuffler. It serves as both a reference implementation
 * and maintains backward compatibility with existing code.
 */
class DefaultShufflerCommunication : public ShufflerCommunicationInterface {
  public:
    /**
     * @brief Constructor for DefaultShufflerCommunication.
     *
     * @param comm The communicator to use for operations.
     * @param op_id The operation ID for tagging messages.
     */
    DefaultShufflerCommunication(std::shared_ptr<Communicator> comm, OpID op_id);

    std::unique_ptr<Communicator::Future> send_chunk_metadata(
        std::unique_ptr<std::vector<uint8_t>> serialized_metadata,
        Rank dest_rank,
        BufferResource* br
    ) override;

    std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank>
    receive_chunk_metadata() override;

    std::unique_ptr<Communicator::Future> send_ready_for_data(
        std::unique_ptr<std::vector<uint8_t>> ready_msg,
        Rank dest_rank,
        BufferResource* br
    ) override;

    std::unique_ptr<Communicator::Future> receive_ready_for_data(
        Rank source_rank, std::unique_ptr<Buffer> buffer
    ) override;

    std::unique_ptr<Communicator::Future> send_gpu_data(
        std::unique_ptr<Buffer> data_buffer, Rank dest_rank
    ) override;

    std::unique_ptr<Communicator::Future> receive_gpu_data(
        Rank source_rank, std::unique_ptr<Buffer> data_buffer
    ) override;

    std::vector<std::unique_ptr<Communicator::Future>> test_some(
        std::vector<std::unique_ptr<Communicator::Future>>& futures
    ) override;

    std::unique_ptr<Buffer> get_gpu_data(
        std::unique_ptr<Communicator::Future> future
    ) override;

  private:
    std::shared_ptr<Communicator> comm_;
    Tag ready_for_data_tag_;
    Tag metadata_tag_;
    Tag gpu_data_tag_;
};

}  // namespace rapidsmpf::shuffler
