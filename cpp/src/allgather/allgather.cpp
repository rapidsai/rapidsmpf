/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include <rapidsmpf/allgather/allgather.hpp>
#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/progress_thread.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::allgather {
namespace detail {

Chunk::Chunk(
    ChunkID id,
    std::unique_ptr<std::vector<std::uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : id_{id},
      metadata_{std::move(metadata)},
      data_{std::move(data)},
      data_size_{data_ ? data_->size : 0} {
    RAPIDSMPF_EXPECTS(metadata_ && data_, "Non-finish chunk must have metadata and data");
}

Chunk::Chunk(ChunkID id) : id_{id}, metadata_{nullptr}, data_{nullptr}, data_size_{0} {}

bool Chunk::is_ready() const noexcept {
    return data_size_ == 0 || (data_ && data_->is_latest_write_done());
}

MemoryType Chunk::memory_type() const noexcept {
    return data_ == nullptr ? MemoryType::HOST : data_->mem_type();
}

bool Chunk::is_finish() const noexcept {
    return data_ == nullptr && metadata_ == nullptr;
}

ChunkID Chunk::id() const noexcept {
    return id_;
}

ChunkID Chunk::sequence() const noexcept {
    return id() & ((static_cast<std::uint64_t>(1) << ID_BITS) - 1);
}

Rank Chunk::origin() const noexcept {
    return id() >> ID_BITS;
}

std::uint64_t Chunk::data_size() const noexcept {
    return data_size_;
}

std::uint64_t Chunk::metadata_size() const noexcept {
    return metadata_ ? metadata_->size() : 0;
}

std::unique_ptr<Chunk> Chunk::from_packed_data(
    std::uint64_t sequence, Rank origin, PackedData&& packed_data
) {
    return std::unique_ptr<Chunk>(new Chunk(
        chunk_id(sequence, origin),
        std::move(packed_data.metadata),
        std::move(packed_data.data)
    ));
}

std::unique_ptr<Chunk> Chunk::from_empty(std::uint64_t sequence, Rank origin) {
    return std::unique_ptr<Chunk>(new Chunk(chunk_id(sequence, origin)));
}

constexpr ChunkID Chunk::chunk_id(std::uint64_t sequence, Rank origin) {
    return (static_cast<std::uint64_t>(origin) << ID_BITS)
           | static_cast<std::uint64_t>(sequence);
}

std::unique_ptr<std::vector<std::uint8_t>> Chunk::serialize() const {
    std::size_t size = sizeof(ChunkID);
    if (!is_finish()) {
        size += sizeof(data_size_) + metadata_size();
    }
    auto result = std::make_unique<std::vector<std::uint8_t>>(size);
    std::memcpy(result->data(), &id_, sizeof(ChunkID));
    if (!is_finish()) {
        std::memcpy(result->data() + sizeof(ChunkID), &data_size_, sizeof(data_size_));
        if (metadata_size() > 0) {
            std::memcpy(
                result->data() + sizeof(ChunkID) + sizeof(data_size_),
                metadata_->data(),
                metadata_->size()
            );
        }
    }
    return result;
}

std::unique_ptr<Chunk> Chunk::deserialize(
    std::vector<std::uint8_t>& data, BufferResource* br
) {
    ChunkID id;
    std::uint64_t data_size;
    std::memcpy(&id, data.data(), sizeof(ChunkID));
    if (data.size() == sizeof(id)) {
        return std::unique_ptr<Chunk>(new Chunk(id));
    }
    std::memcpy(&data_size, data.data() + sizeof(ChunkID), sizeof(data_size));
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        data.size() - sizeof(ChunkID) - sizeof(data_size)
    );
    std::memcpy(
        metadata->data(),
        data.data() + sizeof(ChunkID) + sizeof(data_size),
        metadata->size()
    );
    return std::unique_ptr<Chunk>(new Chunk(
        id,
        std::move(metadata),
        br->allocate(br->stream_pool().get_stream(), br->reserve_or_fail(data_size))
    ));
}

PackedData Chunk::release() {
    RAPIDSMPF_EXPECTS(metadata_ && data_, "Can't release Chunk with no metadata or data");
    return {std::move(metadata_), std::move(data_)};
}

std::unique_ptr<Buffer> Chunk::release_data_buffer() noexcept {
    return std::move(data_);
}

void Chunk::attach_data_buffer(std::unique_ptr<Buffer> data) {
    RAPIDSMPF_EXPECTS(data->size == data_size_, "Mismatching data size");
    RAPIDSMPF_EXPECTS(data_ == nullptr, "Chunk already has data");
    data_ = std::move(data);
}

void PostBox::insert(std::unique_ptr<Chunk> chunk) {
    std::lock_guard lock(mutex_);
    chunks_.emplace_back(std::move(chunk));
}

void PostBox::insert(std::vector<std::unique_ptr<Chunk>>&& chunks) {
    std::lock_guard lock(mutex_);
    std::ranges::for_each(chunks, [&](auto&& chunk) {
        chunks_.emplace_back(std::move(chunk));
    });
}

void PostBox::increment_goalpost(std::uint64_t amount) {
    goalpost_.fetch_add(amount, std::memory_order_acq_rel);
}

bool PostBox::ready() const noexcept {
    std::lock_guard lock(mutex_);
    return goalpost_.load(std::memory_order_acquire) == chunks_.size();
}

std::vector<std::unique_ptr<Chunk>> PostBox::extract_ready() {
    std::lock_guard lock(mutex_);
    std::vector<std::unique_ptr<Chunk>> result;
    for (auto&& chunk : chunks_) {
        if (!chunk->is_ready()) {
            break;
        }
        result.emplace_back(std::move(chunk));
    }
    std::erase(chunks_, nullptr);
    goalpost_.fetch_sub(result.size(), std::memory_order_relaxed);
    return result;
}

std::vector<std::unique_ptr<Chunk>> PostBox::extract() {
    std::lock_guard lock(mutex_);
    goalpost_.fetch_sub(chunks_.size(), std::memory_order_relaxed);
    return std::move(chunks_);
}

bool PostBox::empty() const noexcept {
    std::lock_guard lock(mutex_);
    return chunks_.empty();
}

std::size_t PostBox::spill(
    BufferResource* br, Communicator::Logger& log, std::size_t amount
) {
    std::lock_guard lock(mutex_);
    std::vector<Chunk*> spillable_chunks;
    std::size_t max_spillable{0};
    std::size_t total_spilled{0};
    for (auto&& chunk : chunks_) {
        if (chunk->memory_type() == MemoryType::DEVICE) {
            spillable_chunks.push_back(chunk.get());
            max_spillable += chunk->data_size();
        }
    }
    auto spill_chunk = [&](Chunk* chunk) -> std::size_t {
        auto [reservation, overbooking] =
            br->reserve(MemoryType::HOST, chunk->data_size(), true);
        if (overbooking) {
            log.warn(
                "Cannot spill to host because of host memory overbooking: ",
                format_nbytes(overbooking)
            );
            return 0;
        }
        chunk->attach_data_buffer(br->move(chunk->release_data_buffer(), reservation));
        return chunk->data_size();
    };
    if (max_spillable < amount) {
        // need to spill everything.
        for (auto&& chunk : spillable_chunks) {
            total_spilled += spill_chunk(chunk);
        }
        return total_spilled;
    }
    std::ranges::sort(spillable_chunks, std::less{}, [](Chunk* chunk) {
        return chunk->data_size();
    });
    // Try and spill the minimum number of buffers summing to the
    // amount we need while minimising the amount of data we need to
    // spill.
    while (!spillable_chunks.empty()) {
        auto pos = std::ranges::lower_bound(
            spillable_chunks, amount, std::less{}, [](Chunk* chunk) {
                return chunk->data_size();
            }
        );
        auto chunk = pos == spillable_chunks.end() ? spillable_chunks.back() : *pos;
        total_spilled += spill_chunk(chunk);
        if (total_spilled >= amount) {
            break;
        }
        spillable_chunks.pop_back();
    }
    return total_spilled;
}

static std::vector<std::unique_ptr<Chunk>> test_some(
    std::vector<std::unique_ptr<Chunk>>& chunks,
    std::vector<std::unique_ptr<Communicator::Future>>& futures,
    Communicator* comm
) {
    RAPIDSMPF_EXPECTS(
        chunks.size() == futures.size(), "Mismatching size for chunks and futures"
    );
    if (chunks.empty()) {
        return {};
    }
    auto [complete_futures, indices] = comm->test_some(futures);
    std::vector<std::unique_ptr<Chunk>> result;
    result.reserve(complete_futures.size());
    std::ranges::transform(
        indices, complete_futures, std::back_inserter(result), [&](auto i, auto&& fut) {
            auto chunk = std::move(chunks[i]);
            chunk->attach_data_buffer(comm->release_data(std::move(fut)));
            return std::move(chunk);
        }
    );
    std::erase(chunks, nullptr);
    return result;
}
}  // namespace detail

void AllGather::insert(std::uint64_t sequence_number, PackedData&& packed_data) {
    nlocal_insertions_.fetch_add(1, std::memory_order_relaxed);
    return insert(
        detail::Chunk::from_packed_data(
            sequence_number, comm_->rank(), std::move(packed_data)
        )

    );
}

void AllGather::insert(std::unique_ptr<detail::Chunk> chunk) {
    RAPIDSMPF_EXPECTS(
        !locally_finished_.load(std::memory_order_acquire),
        "Can't insert after locally indicating finished"
    );
    inserted_.insert(std::move(chunk));
}

void AllGather::insert_finished() {
    locally_finished_.store(true, std::memory_order_release);
    inserted_.insert(
        detail::Chunk::from_empty(
            nlocal_insertions_.load(std::memory_order_acquire), comm_->rank()
        )
    );
}

void AllGather::mark_finish(std::uint64_t expected_chunks) noexcept {
    // We must increment the goalpost before decrementing the finish
    // counter so that we cannot, on another thread, observe a finish
    // counter of zero with chunks still to be received.
    for_extraction_.increment_goalpost(expected_chunks);
    finish_counter_.fetch_sub(1, std::memory_order_relaxed);
}

bool AllGather::finished() const noexcept {
    return finish_counter_.load(std::memory_order_acquire) == 0
           && for_extraction_.ready();
}

std::vector<PackedData> AllGather::wait_and_extract(
    AllGather::Ordered ordered, std::chrono::milliseconds timeout
) {
    wait(timeout);
    auto chunks = for_extraction_.extract();
    std::vector<PackedData> result;
    result.reserve(chunks.size());
    if (ordered == AllGather::Ordered::YES) {
        std::ranges::sort(chunks, std::less{}, [](auto&& chunk) { return chunk->id(); });
    }
    std::ranges::transform(chunks, std::back_inserter(result), [](auto&& chunk) {
        return chunk->release();
    });
    return result;
}

std::vector<PackedData> AllGather::extract_ready() {
    // It is OK to extract chunks even if an individual chunk is not
    // ready because the promise is that we deliver data valid in
    // stream-order on the input stream. Even if an output chunk is
    // being spilled, the user must access it in stream order, so we are fine.
    auto chunks = for_extraction_.extract();
    if (chunks.empty()) {
        return {};
    }
    std::vector<PackedData> result;
    result.reserve(chunks.size());
    std::ranges::transform(chunks, std::back_inserter(result), [](auto&& chunk) {
        return chunk->release();
    });
    return result;
}

void AllGather::wait(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (timeout < std::chrono::milliseconds{0}) {
        cv_.wait(lock, [&]() { return can_extract_; });
    } else {
        RAPIDSMPF_EXPECTS(
            cv_.wait_for(lock, timeout, [&]() { return can_extract_; }),
            "wait timeout reached",
            std::runtime_error
        );
    }
}

std::size_t AllGather::spill(std::optional<std::size_t> amount) {
    std::size_t spill_need{0};
    if (amount.has_value()) {
        spill_need = amount.value();
    } else {
        std::int64_t const headroom = br_->memory_available(MemoryType::DEVICE)();
        spill_need = headroom < 0 ? static_cast<std::size_t>(std::abs(headroom)) : 0;
    }
    std::size_t spilled{0};
    if (spill_need > 0) {
        // Spill from ready post box then inserted postbox
        spilled = for_extraction_.spill(br_, comm_->logger(), spill_need);
        if (spilled < spill_need) {
            spilled += inserted_.spill(br_, comm_->logger(), spill_need - spilled);
        }
    }
    return spilled;
}

AllGather::~AllGather() {
    if (active_.load(std::memory_order_acquire)) {
        active_.store(false, std::memory_order_release);
        progress_thread_->remove_function(function_id_);
        br_->spill_manager().remove_spill_function(spill_id_);
    }
}

AllGather::AllGather(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    std::function<void(void)>&& finished_callback
)
    : comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      br_{br},
      statistics_{std::move(statistics)},
      finished_callback_{std::move(finished_callback)},
      finish_counter_{comm_->nranks()},
      op_id_{op_id} {
    function_id_ = progress_thread_->add_function([this]() { return event_loop(); });
    spill_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ -1
    );
}

ProgressThread::ProgressState AllGather::event_loop() {
    /*
     * Data flow:
     * User inserts into inserted_
     * Send side:
     * 1. chunk inserted
     * 2. extract ready chunks
     * 3. for each ready chunk: send metadata to dst and post send for buffer
     * 4. move to chunk to for_extraction_ once send completes
     * 5. chunk is ready for extraction by user
     *
     * Receive side:
     * 1. receive metadata from src
     * 2. allocate chunk and post receive from src
     * 3. Once receive completes
     *  a. If chunk origin is destination (end-of-ring), move to for_extraction_
     *  b. Otherwise insert chunk in inserted_
     *
     * Note: we commit at the point of sending metadata of ready
     * messages to send data immediately. This avoids the need for one
     * ack round-trip.
     */
    Rank const dst = (comm_->rank() + 1) % comm_->nranks();
    Rank const src = (comm_->rank() + comm_->nranks() - 1) % comm_->nranks();
    Tag metadata_tag{op_id_, 0};
    Tag gpu_data_tag{op_id_, 1};
    if (comm_->nranks() == 1) {
        // Note that we don't need to use extract_ready because there is
        // no message passing and our promise to the consumer is that
        // extracted data are valid on the stream used to construct
        // the allgather instance.
        for (auto&& chunk : inserted_.extract()) {
            if (chunk->is_finish()) {
                mark_finish(chunk->sequence());
            } else {
                for_extraction_.insert(std::move(chunk));
            }
        }
    } else {
        // Chunks that are ready to send
        for (auto&& chunk : inserted_.extract_ready()) {
            // Tell the destination about them.
            fire_and_forget_.push_back(
                comm_->send(chunk->serialize(), dst, metadata_tag)
            );
            if (chunk->is_finish()) {
                // Finish chunk contains as sequence number the number
                // of insertions from that rank.
                mark_finish(chunk->sequence());
            } else {
                auto buf = chunk->release_data_buffer();
                sent_posted_.emplace_back(std::move(chunk));
                sent_futures_.emplace_back(
                    comm_->send(std::move(buf), dst, gpu_data_tag)
                );
            }
        }
        while (true) {
            auto const msg = comm_->recv_from(src, metadata_tag);
            if (!msg) {
                break;
            }
            auto chunk = detail::Chunk::deserialize(*msg, br_);
            if (chunk->is_finish()) {
                if (chunk->origin() != dst) {
                    // Finish chunk, if we're not the end of the ring, must forward on.
                    // We will notice this finish when we extract this chunk.
                    inserted_.insert(std::move(chunk));
                } else {
                    // Otherwise, record we're done with data from that rank.
                    mark_finish(chunk->sequence());
                }
            } else {
                // Record we're expecting a chunk.
                to_receive_.emplace_back(std::move(chunk));
            }
        }
        // Post receives if the chunk is ready
        for (auto&& chunk : to_receive_) {
            if (!chunk->is_ready()) {
                break;
            }
            auto buf = chunk->release_data_buffer();
            receive_posted_.emplace_back(std::move(chunk));
            receive_futures_.emplace_back(comm_->recv(src, gpu_data_tag, std::move(buf)));
        }
        std::erase(to_receive_, nullptr);

        std::ranges::for_each(
            detail::test_some(receive_posted_, receive_futures_, comm_.get()),
            [&](auto&& chunk) {
                if (chunk->origin() == dst) {
                    for_extraction_.insert(std::move(chunk));
                } else {
                    inserted_.insert(std::move(chunk));
                }
            }
        );
        for_extraction_.insert(
            detail::test_some(sent_posted_, sent_futures_, comm_.get())
        );
        if (!fire_and_forget_.empty()) {
            std::ignore = comm_->test_some(fire_and_forget_);
        }
    }
    bool const containers_empty =
        (fire_and_forget_.empty() && sent_posted_.empty() && receive_posted_.empty()
         && sent_futures_.empty() && receive_futures_.empty() && to_receive_.empty()
         && inserted_.empty());
    bool const is_finished = finished();
    bool const is_done =
        !active_.load(std::memory_order_acquire) || (is_finished && containers_empty);
    if (is_finished) {
        // We can release our output buffers so notify a waiter.
        {
            std::lock_guard lock(mutex_);
            can_extract_ = true;
        }
        cv_.notify_one();
        std::function<void()> callback = std::move(finished_callback_);
        if (callback) {
            callback();
        }
    }
    return is_done ? ProgressThread::ProgressState::Done
                   : ProgressThread::ProgressState::InProgress;
}

}  // namespace rapidsmpf::allgather
