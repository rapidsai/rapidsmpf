/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda/stream>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/metadata_payload_exchange/core.hpp>
#include <rapidsmpf/communicator/metadata_payload_exchange/tag.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/statistics.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;
using namespace rapidsmpf::communicator;

using AllocationMode = MetadataPayloadExchange::AllocationMode;
using AllocateBufferFn = MetadataPayloadExchange::AllocateBufferFn;

class MetadataPayloadExchangeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        comm = GlobalEnvironment->comm_.get();
        mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        br = BufferResource::create(*mr);
        stream = cuda::stream_ref{cudaStreamLegacy};

        reset_exchange(
            [this](BufferResource*, std::size_t size, AllocationMode) {
                return allocate_receive_buffer(size);
            },
            std::chrono::hours{1}
        );

        GlobalEnvironment->barrier();
    }

    /// Replace the exchange under test with one using the given allocation settings.
    void reset_exchange(
        AllocateBufferFn allocate_fn,
        Duration allocation_timeout,
        OpID op_id = OpID{42},
        BufferResource* exchange_br = nullptr,
        std::optional<std::vector<Rank>> peers = std::nullopt,
        TagMetadataPayloadExchange::ForcedReceive forced_receive =
            TagMetadataPayloadExchange::ForcedReceive::MOVE_TO_DISK
    ) {
        finish_and_wait();
        comm_interface = std::make_unique<TagMetadataPayloadExchange>(
            GlobalEnvironment->comm_,
            op_id,
            peers.value_or(all_remote_ranks(*GlobalEnvironment->comm_)),
            std::move(allocate_fn),
            allocation_timeout,
            exchange_br != nullptr ? exchange_br : br.get(),
            forced_receive
        );
        finished = false;
    }

    /// Call `finish()` on the exchange under test.
    void finish() {
        comm_interface->finish();
        finished = true;
    }

    /// Finish the exchange under test, if any, and progress it until it is idle.
    void finish_and_wait() {
        if (comm_interface == nullptr) {
            return;
        }
        if (!finished) {
            finish();
        }
        auto const deadline = Clock::now() + std::chrono::seconds{30};
        while (!comm_interface->is_idle() && Clock::now() < deadline) {
            comm_interface->progress();
            EXPECT_TRUE(comm_interface->recv().empty()) << "uncollected received message";
            EXPECT_TRUE(comm_interface->release_sent().empty())
                << "uncollected sent message";
            std::this_thread::yield();
        }
        EXPECT_TRUE(comm_interface->is_idle()) << "the exchange did not become idle";
    }

    /// Progress until @p n messages have been received.
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> receive(
        std::size_t n
    ) {
        std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> ret;
        while (ret.size() < n) {
            comm_interface->progress();
            std::ranges::move(comm_interface->recv(), std::back_inserter(ret));
            std::this_thread::yield();
        }
        return ret;
    }

    /// Progress until @p n retained messages have been released, or a deadline passes.
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> release_sent(
        std::size_t n
    ) {
        std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> ret;
        auto const deadline = Clock::now() + std::chrono::seconds{30};
        while (ret.size() < n && Clock::now() < deadline) {
            comm_interface->progress();
            std::ranges::move(comm_interface->release_sent(), std::back_inserter(ret));
            std::this_thread::yield();
        }
        return ret;
    }

    void TearDown() override {
        finish_and_wait();
        GlobalEnvironment->barrier();
    }

    std::unique_ptr<MetadataPayloadExchange::Message> create_test_message(
        Rank peer_rank, std::vector<std::uint8_t> metadata, std::size_t data_size = 0
    ) {
        std::unique_ptr<Buffer> data_buffer = nullptr;
        if (data_size > 0) {
            data_buffer = br->make_buffer(
                stream, br->reserve_or_fail(data_size, MemoryType::DEVICE)
            );
            // Fill with test data
            std::vector<std::uint8_t> test_data(data_size);
            std::iota(test_data.begin(), test_data.end(), 0);
            data_buffer->write_access(
                [&test_data](std::byte* ptr, cuda::stream_ref stream) {
                    RAPIDSMPF_CUDA_TRY(
                        cuda_memcpy_async(ptr, test_data.data(), test_data.size(), stream)
                    );
                }
            );
            data_buffer->stream().sync();
        }

        return std::make_unique<MetadataPayloadExchange::Message>(
            peer_rank, std::move(metadata), std::move(data_buffer)
        );
    }

    /// Create a buffer resource that can spill to disk under @p temp_dir.
    std::shared_ptr<BufferResource> make_disk_br(
        TempDir const& temp_dir,
        std::int64_t disk_limit = std::numeric_limits<std::int64_t>::max()
    ) {
        return BufferResource::create(
            *mr,
            PinnedMemoryDisabled,
            {{MemoryType::DISK, disk_limit}},
            std::nullopt,
            std::make_shared<StreamPool>(1),
            Statistics::disabled(),
            temp_dir.path()
        );
    }

    std::unique_ptr<Buffer> allocate_receive_buffer(std::size_t size) {
        return br->make_buffer(stream, br->reserve_or_fail(size, MemoryType::DEVICE));
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
        RAPIDSMPF_CUDA_TRY(
            cuda_memcpy_async(received_data.data(), buffer->data(), expected_size, stream)
        );
        stream.sync();

        for (std::size_t i = 0; i < expected_size; ++i) {
            EXPECT_EQ(received_data[i], static_cast<std::uint8_t>(i % 256));
        }
    }

    Communicator* comm;
    std::unique_ptr<rmm::mr::cuda_memory_resource> mr;
    cuda::stream_ref stream{cudaStreamLegacy};
    std::shared_ptr<BufferResource> br;
    std::unique_ptr<TagMetadataPayloadExchange> comm_interface;
    bool finished{false};  ///< Whether `finish()` was called on `comm_interface`.
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

    finish_and_wait();
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

    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, RetriesDeferredAllocation) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    constexpr int num_messages = 3;
    constexpr std::size_t num_failures = 10;

    // Fail the first allocations, then succeed. No timeout, so no overbooking. The
    // payload sizes are distinct, so the requested sizes show the allocation order.
    std::vector<std::size_t> requested_sizes;
    int num_overbooked{0};
    reset_exchange(
        [&](BufferResource*,
            std::size_t size,
            AllocationMode mode) -> std::unique_ptr<Buffer> {
            if (mode == AllocationMode::MUST_SUCCEED) {
                ++num_overbooked;
            }
            requested_sizes.push_back(size);
            if (requested_sizes.size() <= num_failures) {
                return nullptr;
            }
            return allocate_receive_buffer(size);
        },
        std::chrono::minutes{10}
    );
    GlobalEnvironment->barrier();

    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    for (int i = 0; i < num_messages; ++i) {
        messages.push_back(create_test_message(
            next_rank, {static_cast<std::uint8_t>(i)}, safe_cast<std::size_t>(i + 1) * 64
        ));
    }
    comm_interface->send(std::move(messages));

    auto received = receive(num_messages);
    ASSERT_EQ(received.size(), num_messages);
    for (int i = 0; i < num_messages; ++i) {
        EXPECT_EQ(received[i]->peer_rank(), prev_rank);
        EXPECT_EQ(received[i]->metadata(), std::vector{static_cast<std::uint8_t>(i)});
        verify_data_content(received[i]->data(), safe_cast<std::size_t>(i + 1) * 64);
    }
    // Buffers are allocated in message order: no later message is attempted before
    // the earlier ones got their buffers.
    EXPECT_GT(requested_sizes.size(), num_failures);
    EXPECT_TRUE(std::ranges::is_sorted(requested_sizes));
    EXPECT_EQ(num_overbooked, 0);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, OverbooksOnTimeout) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    constexpr int num_messages = 3;

    // Allocation only succeeds when overbooking is allowed, so every message has to
    // wait for the timeout and is then overbooked, one at a time, in order.
    int num_overbooked{0};
    reset_exchange(
        [&](BufferResource*,
            std::size_t size,
            AllocationMode mode) -> std::unique_ptr<Buffer> {
            if (mode == AllocationMode::MAY_FAIL) {
                return nullptr;
            }
            ++num_overbooked;
            return allocate_receive_buffer(size);
        },
        std::chrono::milliseconds{10}
    );
    GlobalEnvironment->barrier();

    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    for (int i = 0; i < num_messages; ++i) {
        messages.push_back(
            create_test_message(next_rank, {static_cast<std::uint8_t>(i)}, 128)
        );
    }
    comm_interface->send(std::move(messages));

    auto received = receive(num_messages);
    ASSERT_EQ(received.size(), num_messages);
    for (int i = 0; i < num_messages; ++i) {
        EXPECT_EQ(received[i]->metadata(), std::vector{static_cast<std::uint8_t>(i)});
        verify_data_content(received[i]->data(), 128);
    }
    EXPECT_EQ(num_overbooked, num_messages);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, ThrowsOnTimeoutWithoutOverbooking) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    Rank prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    constexpr std::size_t data_size = 128;
    // Use a separate operation ID, since the payload is drained by hand below.
    OpID const op_id{43};

    // The allocator does not allow forcing an allocation, so it throws when forced.
    int num_forced{0};
    reset_exchange(
        [&](BufferResource*,
            std::size_t,
            AllocationMode mode) -> std::unique_ptr<Buffer> {
            if (mode == AllocationMode::MUST_SUCCEED) {
                ++num_forced;
                throw std::runtime_error("forcing an allocation is not allowed");
            }
            return nullptr;
        },
        std::chrono::milliseconds{10},
        op_id
    );
    GlobalEnvironment->barrier();

    comm_interface->send(create_test_message(next_rank, {0x01}, data_size));
    // Progress until the timeout throws. The deadline only exists to avoid a hang if
    // progress() never throws, so it is far longer than the expected wait.
    bool threw = false;
    auto const deadline = Clock::now() + std::chrono::seconds{30};
    while (!threw && Clock::now() < deadline) {
        try {
            comm_interface->progress();
        } catch (std::runtime_error const&) {
            threw = true;
        }
        std::this_thread::yield();
    }
    EXPECT_TRUE(threw) << "progress() did not throw within the deadline";
    EXPECT_EQ(num_forced, 1);

    // Receive the payload that the exchange gave up on, so it isn't left unmatched.
    auto future = GlobalEnvironment->comm_->recv(
        prev_rank, Tag{op_id, 1}, allocate_receive_buffer(data_size)
    );
    std::ignore = GlobalEnvironment->comm_->wait(std::move(future));
    // Once every rank has received its payload, every send has delivered its data, so
    // destroying the exchange and its send buffers is safe.
    GlobalEnvironment->barrier();
    comm_interface.reset();
}

TEST_F(MetadataPayloadExchangeTest, MovesForcedReceiveToDisk) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    constexpr int num_messages = 3;
    constexpr std::size_t forced_size = 64;

    TempDir temp_dir;
    auto disk_br = make_disk_br(temp_dir);

    // Only the first message has to wait for the timeout, so only its payload is moved
    // to disk. The later messages wait behind it, but their allocations never fail, so
    // they are never forced and stay in device memory.
    reset_exchange(
        [&](BufferResource*,
            std::size_t size,
            AllocationMode mode) -> std::unique_ptr<Buffer> {
            if (size == forced_size && mode == AllocationMode::MAY_FAIL) {
                return nullptr;
            }
            return allocate_receive_buffer(size);
        },
        std::chrono::milliseconds{10},
        OpID{42},
        disk_br.get()
    );
    GlobalEnvironment->barrier();

    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    for (int i = 0; i < num_messages; ++i) {
        messages.push_back(create_test_message(
            next_rank,
            {static_cast<std::uint8_t>(i)},
            forced_size * safe_cast<std::size_t>(i + 1)
        ));
    }
    comm_interface->send(std::move(messages));

    auto received = receive(num_messages);
    ASSERT_EQ(received.size(), num_messages);
    for (int i = 0; i < num_messages; ++i) {
        EXPECT_EQ(received[i]->metadata(), std::vector{static_cast<std::uint8_t>(i)});
    }

    auto on_disk = received[0]->release_data();
    ASSERT_NE(on_disk, nullptr);
    EXPECT_EQ(on_disk->mem_type(), MemoryType::DISK);
    auto host_res = disk_br->reserve_or_fail(forced_size, MemoryType::HOST);
    auto in_memory = disk_br->move(std::move(on_disk), host_res);
    in_memory->stream().sync();
    verify_data_content(in_memory.get(), forced_size);
    for (int i = 1; i < num_messages; ++i) {
        EXPECT_EQ(received[i]->data()->mem_type(), MemoryType::DEVICE);
        verify_data_content(
            received[i]->data(), forced_size * safe_cast<std::size_t>(i + 1)
        );
    }
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, KeepsForcedReceiveInMemoryWithoutDiskSpace) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank next_rank = (comm->rank() + 1) % comm->nranks();
    constexpr std::size_t data_size = 64;

    // No disk space can be reserved, so the forced payload is delivered in memory.
    TempDir temp_dir;
    auto disk_br = make_disk_br(temp_dir, 0);
    reset_exchange(
        [&](BufferResource*,
            std::size_t size,
            AllocationMode mode) -> std::unique_ptr<Buffer> {
            if (mode == AllocationMode::MAY_FAIL) {
                return nullptr;
            }
            return allocate_receive_buffer(size);
        },
        std::chrono::milliseconds{10},
        OpID{42},
        disk_br.get()
    );
    GlobalEnvironment->barrier();

    comm_interface->send(create_test_message(next_rank, {0x01}, data_size));
    auto received = receive(1);
    ASSERT_EQ(received.size(), 1);
    EXPECT_EQ(received[0]->data()->mem_type(), MemoryType::DEVICE);
    verify_data_content(received[0]->data(), data_size);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, SendableMemoryTypes) {
    for (Rank rank = 0; rank < comm->nranks(); ++rank) {
        EXPECT_TRUE(
            std::ranges::equal(
                comm_interface->sendable_memory_types(rank), ADDRESSABLE_MEMORY_TYPES
            )
        );
    }
}

TEST_F(MetadataPayloadExchangeTest, RejectsDiskBackedSend) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank next_rank = (comm->rank() + 1) % comm->nranks();

    TempDir temp_dir;
    auto disk_br = make_disk_br(temp_dir);
    auto data =
        disk_br->make_buffer(stream, disk_br->reserve_or_fail(64, MemoryType::DISK));
    EXPECT_THROW(
        comm_interface->send(
            std::make_unique<MetadataPayloadExchange::Message>(
                next_rank, std::vector<std::uint8_t>{0x01}, std::move(data)
            )
        ),
        std::invalid_argument
    );
    EXPECT_TRUE(comm_interface->is_idle());
}

TEST_F(MetadataPayloadExchangeTest, AllRemoteRanks) {
    auto const ranks = all_remote_ranks(*comm);
    ASSERT_EQ(ranks.size(), safe_cast<std::size_t>(comm->nranks() - 1));
    EXPECT_TRUE(std::ranges::is_sorted(ranks));
    EXPECT_EQ(std::ranges::find(ranks, comm->rank()), ranks.end());
}

TEST_F(MetadataPayloadExchangeTest, RejectsInvalidPeers) {
    auto const make = [&](std::vector<Rank> peers) {
        return TagMetadataPayloadExchange(
            GlobalEnvironment->comm_,
            OpID{44},
            std::move(peers),
            [this](BufferResource*, std::size_t size, AllocationMode) {
                return allocate_receive_buffer(size);
            },
            std::chrono::hours{1},
            br.get()
        );
    };
    EXPECT_THROW(make({comm->rank()}), std::invalid_argument);
    EXPECT_THROW(make({-1}), std::out_of_range);
    EXPECT_THROW(make({comm->nranks()}), std::out_of_range);
    if (comm->nranks() > 1) {
        Rank const next_rank = (comm->rank() + 1) % comm->nranks();
        EXPECT_THROW(make({next_rank, next_rank}), std::invalid_argument);
    }
}

TEST_F(MetadataPayloadExchangeTest, RejectsSendToNonPeer) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank const next_rank = (comm->rank() + 1) % comm->nranks();
    reset_exchange(
        [this](BufferResource*, std::size_t size, AllocationMode) {
            return allocate_receive_buffer(size);
        },
        std::chrono::hours{1},
        OpID{44},
        nullptr,
        std::vector<Rank>{}
    );
    EXPECT_THROW(
        comm_interface->send(create_test_message(next_rank, {0x01}, 64)),
        std::invalid_argument
    );
    EXPECT_TRUE(comm_interface->is_idle());
}

TEST_F(MetadataPayloadExchangeTest, RingPeers) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank const next_rank = (comm->rank() + 1) % comm->nranks();
    Rank const prev_rank = (comm->rank() - 1 + comm->nranks()) % comm->nranks();
    constexpr int num_messages = 3;
    constexpr std::size_t data_size = 256;

    // Messages only go to the next rank, but the previous rank is also a peer, so it
    // takes part in the termination.
    std::vector<Rank> peers{next_rank};
    if (prev_rank != next_rank) {
        peers.push_back(prev_rank);
    }
    reset_exchange(
        [this](BufferResource*, std::size_t size, AllocationMode) {
            return allocate_receive_buffer(size);
        },
        std::chrono::hours{1},
        OpID{45},
        nullptr,
        std::move(peers)
    );
    GlobalEnvironment->barrier();

    for (int i = 0; i < num_messages; ++i) {
        comm_interface->send(
            create_test_message(next_rank, {static_cast<std::uint8_t>(i)}, data_size)
        );
    }
    finish();
    auto received = receive(num_messages);
    ASSERT_EQ(received.size(), num_messages);
    for (int i = 0; i < num_messages; ++i) {
        EXPECT_EQ(received[i]->peer_rank(), prev_rank);
        EXPECT_EQ(received[i]->metadata(), std::vector{static_cast<std::uint8_t>(i)});
        verify_data_content(received[i]->data(), data_size);
    }
    wait_for_communication_complete();
    comm_interface->progress();
    EXPECT_TRUE(comm_interface->recv().empty());
}

TEST_F(MetadataPayloadExchangeTest, SparsePeers) {
    // Ranks only exchange messages with the other ranks of the same parity. With two
    // ranks, every rank has no peers.
    std::vector<Rank> peers;
    for (Rank r = comm->rank() % 2; r < comm->nranks(); r += 2) {
        if (r != comm->rank()) {
            peers.push_back(r);
        }
    }
    reset_exchange(
        [this](BufferResource*, std::size_t size, AllocationMode) {
            return allocate_receive_buffer(size);
        },
        std::chrono::hours{1},
        OpID{46},
        nullptr,
        peers
    );
    GlobalEnvironment->barrier();

    for (Rank peer : peers) {
        comm_interface->send(
            create_test_message(peer, {static_cast<std::uint8_t>(comm->rank())}, 128)
        );
    }
    finish();
    auto received = receive(peers.size());
    ASSERT_EQ(received.size(), peers.size());
    std::vector<Rank> sources;
    for (auto const& msg : received) {
        sources.push_back(msg->peer_rank());
        EXPECT_EQ(
            msg->metadata(), std::vector{static_cast<std::uint8_t>(msg->peer_rank())}
        );
        verify_data_content(msg->data(), 128);
    }
    std::ranges::sort(sources);
    EXPECT_EQ(sources, peers);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, RetainsMessageAfterSend) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank const next_rank = (comm->rank() + 1) % comm->nranks();
    constexpr std::size_t data_size = 512;
    using AfterSend = MetadataPayloadExchange::AfterSend;

    // Messages 0 and 2 have a payload, message 1 has none, and message 3 is
    // discarded.
    auto const make_message =
        [&](std::uint8_t i, std::size_t size, AfterSend after_send) {
            auto msg = create_test_message(next_rank, {i}, size);
            return std::make_unique<MetadataPayloadExchange::Message>(
                next_rank, msg->release_metadata(), msg->release_data(), after_send
            );
        };
    std::vector<std::unique_ptr<MetadataPayloadExchange::Message>> messages;
    messages.push_back(make_message(0, data_size, AfterSend::RETAIN));
    messages.push_back(make_message(1, 0, AfterSend::RETAIN));
    messages.push_back(make_message(2, 2 * data_size, AfterSend::RETAIN));
    messages.push_back(make_message(3, data_size, AfterSend::DISCARD));
    comm_interface->send(std::move(messages));

    auto received = receive(4);
    auto sent = release_sent(3);
    ASSERT_EQ(sent.size(), 3);
    for (std::uint8_t i = 0; i < 3; ++i) {
        EXPECT_EQ(sent[i]->peer_rank(), next_rank);
        EXPECT_EQ(sent[i]->metadata(), std::vector{i});
    }
    verify_data_content(sent[0]->data(), data_size);
    EXPECT_EQ(sent[1]->data(), nullptr);
    verify_data_content(sent[2]->data(), 2 * data_size);

    ASSERT_EQ(received.size(), 4);
    for (std::uint8_t i = 0; i < 4; ++i) {
        EXPECT_EQ(received[i]->metadata(), std::vector{i});
    }
    wait_for_communication_complete();
    EXPECT_TRUE(comm_interface->release_sent().empty());
}

TEST_F(MetadataPayloadExchangeTest, IsNotIdleUntilReceivedMessageIsReturned) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank const next_rank = (comm->rank() + 1) % comm->nranks();

    comm_interface->send(create_test_message(next_rank, {0x01}, 64));
    finish();
    // After finish(), the exchange is busy until the message has arrived, and then
    // until recv() has returned it.
    for (int i = 0; i < 100; ++i) {
        comm_interface->progress();
        ASSERT_FALSE(comm_interface->is_idle());
        std::this_thread::yield();
    }
    auto received = receive(1);
    ASSERT_EQ(received.size(), 1);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, IsNotIdleUntilRetainedMessageIsReleased) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank const next_rank = (comm->rank() + 1) % comm->nranks();

    auto msg = create_test_message(next_rank, {0x01}, 64);
    comm_interface->send(
        std::make_unique<MetadataPayloadExchange::Message>(
            next_rank,
            msg->release_metadata(),
            msg->release_data(),
            MetadataPayloadExchange::AfterSend::RETAIN
        )
    );
    auto received = receive(1);
    // Once every rank has received its message, every send has delivered its data,
    // so progress() detects the completed send. The exchange must stay busy until
    // the retained message is released.
    GlobalEnvironment->barrier();
    for (int i = 0; i < 100; ++i) {
        comm_interface->progress();
        ASSERT_FALSE(comm_interface->is_idle());
        std::this_thread::yield();
    }
    auto sent = release_sent(1);
    ASSERT_EQ(sent.size(), 1);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, KeepsForcedReceiveInMemoryWhenDiskMoveDisabled) {
    if (comm->nranks() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }
    Rank const next_rank = (comm->rank() + 1) % comm->nranks();
    constexpr std::size_t data_size = 64;

    TempDir temp_dir;
    auto disk_br = make_disk_br(temp_dir);
    reset_exchange(
        [&](BufferResource*,
            std::size_t size,
            AllocationMode mode) -> std::unique_ptr<Buffer> {
            if (mode == AllocationMode::MAY_FAIL) {
                return nullptr;
            }
            return allocate_receive_buffer(size);
        },
        std::chrono::milliseconds{10},
        OpID{42},
        disk_br.get(),
        std::nullopt,
        TagMetadataPayloadExchange::ForcedReceive::KEEP_IN_MEMORY
    );
    GlobalEnvironment->barrier();

    comm_interface->send(create_test_message(next_rank, {0x01}, data_size));
    auto received = receive(1);
    ASSERT_EQ(received.size(), 1);
    EXPECT_EQ(received[0]->data()->mem_type(), MemoryType::DEVICE);
    verify_data_content(received[0]->data(), data_size);
    finish_and_wait();
}

TEST_F(MetadataPayloadExchangeTest, DefaultAllocateBuffer) {
    // With memory available, the allocation is made in device memory.
    auto buffer = default_allocate_buffer(br.get(), 128, AllocationMode::MAY_FAIL);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->size, 128);
    EXPECT_EQ(buffer->mem_type(), MemoryType::DEVICE);

    // Without memory available, the allocation fails unless it must succeed, in which
    // case host memory is overbooked.
    auto const make_full_br = [&](AllowOverbooking allow_overbooking) {
        return BufferResource::create(
            *mr,
            PinnedMemoryDisabled,
            {{MemoryType::DEVICE, 0}, {MemoryType::HOST, 0}},
            std::nullopt,
            std::make_shared<StreamPool>(1),
            Statistics::disabled(),
            std::nullopt,
            std::chrono::milliseconds{100},
            allow_overbooking
        );
    };
    auto full_br = make_full_br(AllowOverbooking::YES);
    EXPECT_EQ(
        default_allocate_buffer(full_br.get(), 128, AllocationMode::MAY_FAIL), nullptr
    );
    buffer = default_allocate_buffer(full_br.get(), 128, AllocationMode::MUST_SUCCEED);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->size, 128);
    EXPECT_EQ(buffer->mem_type(), MemoryType::HOST);

    auto strict_br = make_full_br(AllowOverbooking::NO);
    EXPECT_EQ(
        default_allocate_buffer(strict_br.get(), 128, AllocationMode::MAY_FAIL), nullptr
    );
    EXPECT_THROW(
        std::ignore =
            default_allocate_buffer(strict_br.get(), 128, AllocationMode::MUST_SUCCEED),
        std::runtime_error
    );
}
