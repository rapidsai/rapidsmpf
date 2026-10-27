/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::communicator {

/**
 * @brief Interface for exchanging serialized metadata and payload between ranks.
 *
 * The `MetadataPayloadExchange` class defines an abstract interface for transmitting
 * messages that contain both serialized metadata and a data payload. This abstraction
 * simplifies scenarios where metadata and payload must be exchanged together as a
 * single logical unit.
 *
 * Concrete implementations, such as `TagMetadataPayloadExchange`, use the
 * `Communicator` to implement this interface. In the future, other implementations
 * may leverage specialized features beyond the basic `Communicator` API to further
 * optimize this communication pattern.
 *
 * @note This class is not thread-safe. All methods except `sendable_memory_types()` must
 * be called from the same thread.
 *
 * @note All concrete implementations are expected to provide a constructor with
 * the following signature:
 * @code
 * DerivedMetadataPayloadExchange(
 *     std::shared_ptr<Communicator> comm,
 *     OpID op_id,
 *     std::vector<Rank> peers,
 *     MetadataPayloadExchange::AllocateBufferFn allocate_buffer_fn,
 *     Duration allocation_timeout,
 *     BufferResource* br
 * );
 * @endcode
 *
 * The exchange only communicates with its `peers`: messages can only be sent to a
 * peer, are only received from peers, and `finish()` and `is_idle()` only involve
 * peers. The peers must not include the calling rank or duplicates, and the peer
 * relation must be symmetric: if rank A has rank B as a peer, rank B has rank A as a
 * peer. This is not checked: if rank B does not have rank A as a peer, rank A waits
 * forever for the termination marker of rank B and never becomes idle, and messages
 * from rank A stay unreceived on rank B, where a later exchange reusing the operation
 * ID may receive them. Use `all_remote_ranks()` to communicate with every other rank.
 *
 * An exchange is used in three steps:
 * 1. Send messages with `send()`, and call `progress()`, `recv()` and `release_sent()`
 *    regularly.
 * 2. Call `finish()` once no more messages will be sent.
 * 3. Keep calling `progress()`, `recv()` and `release_sent()` until `is_idle()` returns
 *    `true`.
 *
 * Only then can the exchange be destroyed and its operation ID be reused.
 *
 * Receive buffers are allocated with `allocate_buffer_fn`, which is called on the
 * thread calling `progress()` and must not block or spill. Implementations follow
 * these rules:
 * - When called with `AllocationMode::MAY_FAIL`, the allocator may return `nullptr` to
 *   indicate that memory is not available right now. The message is kept and the
 *   allocation is retried on later calls to `progress()`. A message is never dropped
 *   or delivered without its payload.
 * - If no allocation succeeds within `allocation_timeout`, one pending message is
 *   forced: the allocator is called with `AllocationMode::MUST_SUCCEED` and must
 *   return a buffer or throw. An exception thrown by the allocator propagates out of
 *   `progress()` and is a fatal, unrecoverable error: the exchange can't be used
 *   afterwards. Only one message is forced per timeout. The timeout measures time
 *   without any successful allocation, not how long a single message has waited, so
 *   a message can wait longer while other allocations keep succeeding.
 * - The order in which pending allocations are retried is implementation defined.
 *
 * Payloads at rest may live on disk:
 * - When `br` has a disk resource, implementations may move the payload of a message
 *   whose receive buffer was forced by the timeout to disk before the message is
 *   delivered, so received messages may be disk-backed. The move blocks `progress()`
 *   while the payload is written. Without a disk resource, the payload is delivered in
 *   the forced buffer.
 * - Only payloads in `sendable_memory_types()` of their destination can be sent.
 */
class MetadataPayloadExchange {
  public:
    /**
     * @brief Whether a receive buffer allocation may fail.
     *
     * See `AllocateBufferFn`.
     */
    enum class AllocationMode : bool {
        MAY_FAIL,  ///< The allocator may return `nullptr` if memory is not available now.
        MUST_SUCCEED,  ///< The allocator must return a buffer or throw.
    };

    /**
     * @brief Function allocating a receive buffer of the given size.
     *
     * The exchange calls it with its own buffer resource.
     *
     * With `AllocationMode::MAY_FAIL`, it may return `nullptr` if memory is not available
     * right now. With `AllocationMode::MUST_SUCCEED`, it must return a buffer, for
     * example by overbooking memory, or throw if its policy does not allow forcing an
     * allocation. Throwing is a fatal, unrecoverable error, so only throw as the very
     * last resort.
     */
    using AllocateBufferFn = std::function<
        std::unique_ptr<Buffer>(BufferResource*, std::size_t, AllocationMode)>;

    /**
     * @brief What happens to an outgoing message once it has been sent.
     */
    enum class AfterSend : bool {
        DISCARD,  ///< The message is discarded.
        RETAIN,  ///< The message is retained and returned by `release_sent()`.
    };

    /**
     * @brief Message class for communication.
     *
     * This class contains the essential information needed for communication:
     * data payload, metadata, and peer rank (source/destination).
     */
    class Message {
      public:
        /**
         * @brief Construct a new Message.
         *
         * @param peer_rank Destination (outgoing) or source (incoming) rank.
         * @param metadata Serialized metadata.
         * @param data Data buffer (can be nullptr for metadata-only messages).
         * @param after_send What happens to an outgoing message once it has been sent.
         */
        Message(
            Rank peer_rank,
            std::vector<std::uint8_t>&& metadata,
            std::unique_ptr<Buffer> data = nullptr,
            AfterSend after_send = AfterSend::DISCARD
        );

        /**
         * @brief Get the destination rank for outgoing or source rank for incoming
         * messages.
         *
         * @return The rank of the destination or source.
         */
        [[nodiscard]] constexpr Rank peer_rank() const noexcept {
            return peer_rank_;
        }

        /**
         * @brief Get what happens to an outgoing message once it has been sent.
         *
         * @return Whether the message is discarded or retained for `release_sent()`.
         */
        [[nodiscard]] constexpr AfterSend after_send() const noexcept {
            return after_send_;
        }

        /**
         * @brief Get the serialized metadata for this message.
         *
         * This metadata is sent first to inform the receiver about the incoming message.
         *
         * @return The serialized metadata.
         */
        [[nodiscard]] constexpr std::vector<std::uint8_t> const&
        metadata() const noexcept {
            return metadata_;
        }

        /**
         * @brief Release ownership of the metadata.
         *
         * This is typically called when transferring metadata to the communication layer.
         *
         * @return Metadata with ownership transferred.
         */
        [[nodiscard]] std::vector<std::uint8_t> release_metadata() noexcept;

        /**
         * @brief Get the data buffer for this message.
         *
         * @return The data buffer, or nullptr if no data.
         */
        [[nodiscard]] Buffer const* data() const;

        /**
         * @brief Release ownership of the data buffer.
         *
         * This is typically called when transferring a buffer to the communication layer.
         *
         * @return Data buffer with ownership transferred, or nullptr if no data.
         */
        [[nodiscard]] std::unique_ptr<Buffer> release_data() noexcept;

        /**
         * @brief Set the data buffer for this message.
         *
         * This method can be used by implementations to update the data buffer.
         *
         * @param buffer Data buffer to be set.
         */
        void set_data(std::unique_ptr<Buffer> buffer);

      private:
        Rank peer_rank_;
        std::vector<std::uint8_t> metadata_;
        std::unique_ptr<Buffer> data_;
        AfterSend after_send_;
    };

    virtual ~MetadataPayloadExchange() = default;

    /**
     * @brief Get the memory types of payloads that can be sent to a remote rank.
     *
     * Callers must only send payloads in one of these memory types. The result
     * includes at least `ADDRESSABLE_MEMORY_TYPES`, is fixed at construction, and this
     * method may be called from any thread.
     *
     * @param dst The destination rank.
     * @return The sendable memory types in decreasing order of preference.
     */
    [[nodiscard]] virtual std::span<MemoryType const> sendable_memory_types(
        Rank dst
    ) const = 0;

    /**
     * @brief Send a single message to a remote rank.
     *
     * Takes ownership of a ready message and manages its transmission, including
     * metadata sending and coordination of data transfer.
     *
     * The messages sent from the calling process to a destination remote rank are
     * guaranteed to be received in the same order as they were sent. No ordering is
     * guaranteed between messages sent to different remote ranks.
     *
     * @param message Message ready to be sent to a remote rank.
     */
    virtual void send(std::unique_ptr<Message> message) = 0;

    /**
     * @brief Send messages to remote ranks.
     *
     * Takes ownership of ready messages and manages their transmission, including
     * metadata sending and coordination of data transfer.
     *
     * The messages sent from the calling process to a destination remote rank are
     * guaranteed to be received in the same order as they were sent. No ordering is
     * guaranteed between messages sent to different remote ranks.
     *
     * @param messages Vector of messages ready to be sent to remote ranks.
     */
    virtual void send(std::vector<std::unique_ptr<Message>>&& messages) = 0;

    /**
     * @brief Progress the communication state machine.
     *
     * Advances the internal state of the communication layer by processing pending
     * operations such as receiving metadata, setting up data transfers, completing
     * data transfers, and cleaning up completed operations. Completed messages are
     * stored internally and can be retrieved via recv().
     *
     * This method should be called periodically to make progress on communication.
     */
    virtual void progress() = 0;

    /**
     * @brief Receive messages from remote ranks.
     *
     * The messages received by the calling process are guaranteed to be received in the
     * same order as they were sent by the source remote rank. No ordering is guaranteed
     * between messages received from different remote ranks.
     *
     * @return Vector of completed messages ready for local processing.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Message>> recv() = 0;

    /**
     * @brief Release retained messages whose send has completed.
     *
     * Returns the messages sent with `AfterSend::RETAIN` whose send has completed, as
     * detected by `progress()`. A released message is unchanged: it has its peer rank,
     * metadata and payload, which can be reused. Messages to the same destination rank
     * are returned in the order they were sent. No ordering is guaranteed between
     * messages sent to different ranks.
     *
     * @return Vector of sent messages.
     */
    [[nodiscard]] virtual std::vector<std::unique_ptr<Message>> release_sent() = 0;

    /**
     * @brief Signal that no more messages will be sent.
     *
     * After calling this method, no further calls to send() are permitted.
     * The implementation sends protocol-level termination markers to all peers so
     * that each receiver knows the exact number of application messages to expect.
     * This enables safe reuse of operation IDs: once all termination markers have
     * been received and all expected messages processed, the communication layer
     * considers itself idle and the tag/op_id can be reused.
     *
     * @throws std::logic_error If called more than once.
     */
    virtual void finish() = 0;

    /**
     * @brief Check if the communication layer is currently idle.
     *
     * Indicates whether there are any active or pending communication operations.
     * Before finish() is called, a return value of `true` means no I/O operations
     * are in progress, and no messages wait to be returned by `recv()` or
     * `release_sent()`. After finish() is called, `true` additionally requires that
     * all peers have sent their termination markers and all expected messages have
     * been received, meaning the op_id can safely be reused.
     *
     * @return `true` if the communication layer is idle; `false` if activity is ongoing.
     */
    [[nodiscard]] virtual bool is_idle() const = 0;
};

/**
 * @brief Get all ranks of a communicator except the calling rank.
 *
 * Use as the peers of a `MetadataPayloadExchange` that communicates with every other
 * rank.
 *
 * @param comm The communicator.
 * @return The ranks in increasing order, excluding `comm.rank()`.
 */
[[nodiscard]] std::vector<Rank> all_remote_ranks(Communicator const& comm);

/**
 * @brief The default receive buffer allocator of a `MetadataPayloadExchange`.
 *
 * See `MetadataPayloadExchange::AllocateBufferFn`. Reserves memory in
 * `ADDRESSABLE_MEMORY_TYPES`, in order of preference, without overbooking. If nothing
 * fits:
 * - With `AllocationMode::MAY_FAIL`, it returns `nullptr`.
 * - With `AllocationMode::MUST_SUCCEED`, it overbooks host memory if
 *   `allow_overbooking_by_default()` of @p br allows it, and otherwise throws
 *   `std::runtime_error`, which is a fatal, unrecoverable error for the exchange.
 *
 * @param br The buffer resource to allocate from.
 * @param size The size of the buffer in bytes.
 * @param mode Whether the allocation may fail.
 * @return The buffer, or `nullptr` if @p mode allows failing and nothing fits.
 *
 * @throws std::runtime_error If the allocation must succeed, nothing fits, and
 * overbooking is not allowed.
 */
[[nodiscard]] std::unique_ptr<Buffer> default_allocate_buffer(
    BufferResource* br, std::size_t size, MetadataPayloadExchange::AllocationMode mode
);

}  // namespace rapidsmpf::communicator
