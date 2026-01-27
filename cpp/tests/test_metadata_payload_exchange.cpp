/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::communicator;

class MetadataPayloadExchangeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        comm = GlobalEnvironment->comm_.get();
        mr = std::unique_ptr<rmm::mr::device_memory_resource>(
            new rmm::mr::cuda_memory_resource{}
        );
        br = std::make_unique<BufferResource>(mr.get());
        stream = rmm::cuda_stream_default;
        statistics = std::make_shared<Statistics>();

        auto allocate_fn = [this](std::size_t size) {
            return allocate_receive_buffer(size);
        };

        comm_interface = std::make_unique<TagMetadataPayloadExchange>(
            GlobalEnvironment->comm_, OpID{42}, allocate_fn, statistics
        );

        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

    std::unique_ptr<MetadataPayloadExchange::Message> create_test_message(
        Rank peer_rank, std::vector<std::uint8_t> metadata, std::size_t data_size = 0
    ) {
        std::unique_ptr<Buffer> data_buffer = nullptr;
        if (data_size > 0) {
            data_buffer =
                br->allocate(stream, br->reserve_or_fail(data_size, MemoryType::DEVICE));
            // Fill with test data
            data_buffer->write_access(
                [data_size](std::byte* ptr, rmm::cuda_stream_view stream) {
                    std::vector<std::uint8_t> test_data(data_size);
                    std::iota(test_data.begin(), test_data.end(), 0);
                    RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                        ptr, test_data.data(), data_size, cudaMemcpyHostToDevice, stream
                    ));
                }
            );
            data_buffer->stream().synchronize();
        }

        return std::make_unique<MetadataPayloadExchange::Message>(
            peer_rank, std::move(metadata), std::move(data_buffer)
        );
    }

    std::unique_ptr<Buffer> allocate_receive_buffer(std::size_t size) {
        return br->allocate(stream, br->reserve_or_fail(size, MemoryType::DEVICE));
    }

    void wait_for_communication_complete() {
        while (!comm_interface->is_idle()) {
            comm_interface->progress();
            std::this_thread::yield();
        }
    }

    void verify_data_content(Buffer const* buffer, std::size_t expected_size) {
        ASSERT_NE(buffer, nullptr);
        EXPECT_EQ(buffer->size, expected_size);

        std::vector<std::uint8_t> received_data(expected_size);
        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
            received_data.data(),
            buffer->data(),
            expected_size,
            cudaMemcpyDeviceToHost,
            stream
        ));
        stream.synchronize();

        for (std::size_t i = 0; i < expected_size; ++i) {
            EXPECT_EQ(received_data[i], static_cast<std::uint8_t>(i % 256));
        }
    }

    Communicator* comm;
    std::unique_ptr<rmm::mr::device_memory_resource> mr;
    rmm::cuda_stream_view stream;
    std::unique_ptr<BufferResource> br;
    std::shared_ptr<Statistics> statistics;
    std::unique_ptr<TagMetadataPayloadExchange> comm_interface;
};

TEST_F(MetadataPayloadExchangeTest, InitialState) {
    // Communication interface should start in idle state
    EXPECT_TRUE(comm_interface->is_idle());
}

TEST_F(MetadataPayloadExchangeTest, BroadcastData) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::vector<std::uint8_t> test_metadata = {0xDE, 0xAD, 0xBE, 0xEF};
    constexpr std::size_t data_size = 2048;

    if (comm->rank() == 0) {
        // Rank 0 sends to all other ranks
        std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
        for (int r = 1; r < comm->nranks(); ++r) {
            messages.push_back(create_test_message(Rank{r}, test_metadata, data_size));
        }

        EXPECT_TRUE(comm_interface->is_idle());
        comm_interface->send(std::move(messages));
        EXPECT_FALSE(comm_interface->is_idle());

        // Rank 0 should not receive any messages
        for (int i = 0; i < 5; ++i) {
            comm_interface->progress();
            auto recvd = comm_interface->recv();
            EXPECT_TRUE(recvd.empty());
            if (comm_interface->is_idle())
                break;
            std::this_thread::yield();
        }

        wait_for_communication_complete();
        EXPECT_TRUE(comm_interface->is_idle());
    } else {
        // Other ranks only receive one message from rank 0
        std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> received_messages;
        while (received_messages.empty()) {
            comm_interface->progress();
            auto messages = comm_interface->recv();
            std::ranges::move(messages, std::back_inserter(received_messages));

            if (received_messages.empty()) {
                std::this_thread::yield();
            }
        }

        EXPECT_EQ(received_messages.size(), 1);
        auto& msg = received_messages[0];
        EXPECT_EQ(msg->peer_rank(), Rank{0});
        EXPECT_EQ(msg->metadata(), test_metadata);
        ASSERT_NE(msg->data(), nullptr);
        verify_data_content(msg->data(), data_size);

        wait_for_communication_complete();
        EXPECT_TRUE(comm_interface->is_idle());
    }
}

TEST_F(MetadataPayloadExchangeTest, SetDataOnAlreadySetData) {
    // Test that set_data throws when data is already set
    std::vector<std::uint8_t> test_metadata = {0x01};
    constexpr std::size_t data_size = 256;

    // Create a message with data already set
    auto message = create_test_message(Rank{0}, test_metadata, data_size);

    // Try to set data again - should throw
    auto buffer = allocate_receive_buffer(128);
    EXPECT_THROW(message->set_data(std::move(buffer)), std::logic_error);
}

TEST_F(MetadataPayloadExchangeTest, SendReceiveMetadataOnly) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    std::vector<std::uint8_t> test_metadata = {0x01, 0x02, 0x03, 0x04};

    // All ranks send metadata-only message to next rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    messages.push_back(create_test_message(next_rank, test_metadata));

    EXPECT_TRUE(comm_interface->is_idle());
    comm_interface->send(std::move(messages));
    EXPECT_FALSE(comm_interface->is_idle());

    // All ranks receive from previous rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> received_messages;
    while (received_messages.empty()) {
        comm_interface->progress();
        auto messages = comm_interface->recv();
        std::ranges::move(messages, std::back_inserter(received_messages));

        if (received_messages.empty()) {
            std::this_thread::yield();
        }
    }

    EXPECT_EQ(received_messages.size(), 1);
    auto& msg = received_messages[0];
    EXPECT_EQ(msg->peer_rank(), prev_rank);
    EXPECT_EQ(msg->metadata(), test_metadata);
    EXPECT_EQ(msg->data(), nullptr);

    wait_for_communication_complete();

    EXPECT_TRUE(comm_interface->is_idle());
}

TEST_F(MetadataPayloadExchangeTest, SendReceiveZeroSizedMetadataOnly) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    std::vector<std::uint8_t> test_metadata = {};

    // All ranks send metadata-only message with empty metadata to next rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    messages.push_back(create_test_message(next_rank, test_metadata));

    EXPECT_TRUE(comm_interface->is_idle());
    comm_interface->send(std::move(messages));
    EXPECT_FALSE(comm_interface->is_idle());

    // All ranks receive from previous rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> received_messages;
    while (received_messages.empty()) {
        comm_interface->progress();
        auto messages = comm_interface->recv();
        std::ranges::move(messages, std::back_inserter(received_messages));

        if (received_messages.empty()) {
            std::this_thread::yield();
        }
    }

    EXPECT_EQ(received_messages.size(), 1);
    auto& msg = received_messages[0];
    EXPECT_EQ(msg->peer_rank(), prev_rank);
    EXPECT_TRUE(msg->metadata().empty());
    EXPECT_EQ(msg->metadata(), test_metadata);
    EXPECT_EQ(msg->data(), nullptr);

    wait_for_communication_complete();

    EXPECT_TRUE(comm_interface->is_idle());
}

TEST_F(MetadataPayloadExchangeTest, SendReceiveSingleMessage) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    std::vector<std::uint8_t> test_metadata = {0xAA, 0xBB, 0xCC, 0xDD};
    constexpr std::size_t data_size = 512;

    // All ranks send single message to next rank using send method
    auto message = create_test_message(next_rank, test_metadata, data_size);

    EXPECT_TRUE(comm_interface->is_idle());
    comm_interface->send(std::move(message));
    EXPECT_FALSE(comm_interface->is_idle());

    // All ranks receive from previous rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> received_messages;
    while (received_messages.empty()) {
        comm_interface->progress();
        auto messages = comm_interface->recv();
        std::ranges::move(messages, std::back_inserter(received_messages));

        if (received_messages.empty()) {
            std::this_thread::yield();
        }
    }

    EXPECT_EQ(received_messages.size(), 1);
    auto& msg = received_messages[0];
    EXPECT_EQ(msg->peer_rank(), prev_rank);
    EXPECT_EQ(msg->metadata(), test_metadata);
    verify_data_content(msg->data(), data_size);

    wait_for_communication_complete();

    EXPECT_TRUE(comm_interface->is_idle());
}

TEST_F(MetadataPayloadExchangeTest, SendReceiveWithData) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    std::vector<std::uint8_t> test_metadata = {0x10, 0x20, 0x30, 0x40};
    constexpr std::size_t data_size = 1024;

    // All ranks send message with data to next rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    messages.push_back(create_test_message(next_rank, test_metadata, data_size));

    comm_interface->send(std::move(messages));

    // All ranks receive from previous rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> received_messages;
    while (received_messages.empty()) {
        comm_interface->progress();
        auto messages = comm_interface->recv();
        std::ranges::move(messages, std::back_inserter(received_messages));

        if (received_messages.empty()) {
            std::this_thread::yield();
        }
    }

    EXPECT_EQ(received_messages.size(), 1);
    auto& msg = received_messages[0];
    EXPECT_EQ(msg->peer_rank(), prev_rank);
    EXPECT_EQ(msg->metadata(), test_metadata);
    verify_data_content(msg->data(), data_size);

    wait_for_communication_complete();
}

TEST_F(MetadataPayloadExchangeTest, MultipleMessages) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    constexpr int num_messages = 5;

    // All ranks send multiple messages to next rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;

    for (int i = 0; i < num_messages; ++i) {
        std::vector<std::uint8_t> metadata = {
            static_cast<std::uint8_t>(i),
            static_cast<std::uint8_t>(i + 1),
            static_cast<std::uint8_t>(i + 2)
        };
        std::size_t data_size =
            (i % 2 == 0)
                ? 0
                : (i + 1) * 100;  // Alternate between metadata-only and with-data
        messages.push_back(create_test_message(next_rank, metadata, data_size));
    }

    comm_interface->send(std::move(messages));

    // All ranks receive from previous rank
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> received_messages;
    while (received_messages.size() < num_messages) {
        comm_interface->progress();
        auto messages = comm_interface->recv();
        std::ranges::move(messages, std::back_inserter(received_messages));

        if (received_messages.empty()) {
            std::this_thread::yield();
        }
    }

    // All ranks should get all messages from previous rank
    EXPECT_EQ(received_messages.size(), num_messages);

    for (int i = 0; i < static_cast<int>(received_messages.size()); ++i) {
        auto& msg = received_messages[i];
        EXPECT_EQ(msg->peer_rank(), prev_rank);

        // Check metadata
        std::vector<std::uint8_t> expected_metadata = {
            static_cast<std::uint8_t>(i),
            static_cast<std::uint8_t>(i + 1),
            static_cast<std::uint8_t>(i + 2)
        };
        EXPECT_EQ(msg->metadata(), expected_metadata);

        // Check data presence and content
        if (i % 2 == 0) {
            EXPECT_EQ(msg->data(), nullptr);  // Even indices are metadata-only
        } else {
            EXPECT_NE(msg->data(), nullptr);  // Odd indices have data
            std::size_t expected_size = (i + 1) * 100;
            verify_data_content(msg->data(), expected_size);  // Odd indices have data
        }
    }

    wait_for_communication_complete();
}
