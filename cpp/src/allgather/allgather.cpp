/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>

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
      data_size_{data_ ? data_->size : 0},
      is_finish_{false} {
    RAPIDSMPF_EXPECTS(metadata_ && data_, "Non-finish chunk must have metadata and data");
}

Chunk::Chunk(ChunkID id)
    : id_{id}, metadata_{nullptr}, data_{nullptr}, data_size_{0}, is_finish_{true} {}

bool Chunk::is_ready() const noexcept {
    return data_size_ == 0 || (data_ && data_->is_ready());
}

MemoryType Chunk::memory_type() const noexcept {
    return data_ == nullptr ? MemoryType::HOST : data_->mem_type();
}

bool Chunk::is_finish() const noexcept {
    return is_finish_;
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
    ChunkID sequence, Rank origin, PackedData&& packed_data
) {
    ChunkID const id = (static_cast<std::uint64_t>(origin) << ID_BITS)
                       | static_cast<std::uint64_t>(sequence);
    return std::unique_ptr<Chunk>(
        new Chunk(id, std::move(packed_data.metadata), std::move(packed_data.data))
    );
}

std::unique_ptr<Chunk> Chunk::from_empty(ChunkID sequence, Rank origin) {
    ChunkID const id = (static_cast<std::uint64_t>(origin) << ID_BITS)
                       | static_cast<std::uint64_t>(sequence);
    return std::unique_ptr<Chunk>(new Chunk(id));
}

std::unique_ptr<std::vector<std::uint8_t>> Chunk::serialize() const {
    auto result = std::make_unique<std::vector<std::uint8_t>>(
        sizeof(ChunkID) + sizeof(data_size_) + sizeof(is_finish_) + metadata_size()
    );
    std::memcpy(result->data(), &id_, sizeof(ChunkID));
    std::memcpy(result->data() + sizeof(ChunkID), &data_size_, sizeof(data_size_));
    std::memcpy(
        result->data() + sizeof(ChunkID) + sizeof(data_size_),
        &is_finish_,
        sizeof(is_finish_)
    );
    if (metadata_size() > 0) {
        std::memcpy(
            result->data() + sizeof(ChunkID) + sizeof(data_size_) + sizeof(is_finish_),
            metadata_->data(),
            metadata_->size()
        );
    }
    return result;
}

std::unique_ptr<Chunk> Chunk::deserialize(
    std::vector<std::uint8_t>& data, rmm::cuda_stream_view stream, BufferResource* br
) {
    ChunkID id;
    std::uint64_t data_size;
    bool is_finish;
    std::memcpy(&id, data.data(), sizeof(ChunkID));
    std::memcpy(&data_size, data.data() + sizeof(ChunkID), sizeof(data_size));
    std::memcpy(
        &is_finish, data.data() + sizeof(ChunkID) + sizeof(data_size), sizeof(is_finish)
    );
    if (is_finish) {
        return std::unique_ptr<Chunk>(new Chunk(id));
    }
    auto metadata = [&]() -> std::unique_ptr<std::vector<std::uint8_t>> {
        auto size = data.size() - sizeof(ChunkID) - sizeof(data_size) - sizeof(is_finish);
        auto metadata = std::make_unique<std::vector<std::uint8_t>>(size);
        std::memcpy(
            metadata->data(),
            data.data() + sizeof(ChunkID) + sizeof(data_size) + sizeof(is_finish),
            size
        );
        return metadata;
    }();
    auto [r, _] = br->reserve(MemoryType::DEVICE, data_size, false);
    if (r.size() != data_size) {
        std::tie(r, std::ignore) = br->reserve(MemoryType::HOST, data_size, false);
        RAPIDSMPF_EXPECTS(r.size() == data_size, "Can't reserve device or host memory");
    }
    return std::unique_ptr<Chunk>(
        new Chunk(id, std::move(metadata), br->allocate(data_size, stream, r))
    );
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
    goalpost_.fetch_add(amount, std::memory_order_relaxed);
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
    std::vector<std::unique_ptr<Chunk>> result;
    std::swap(chunks_, result);
    goalpost_.fetch_sub(result.size(), std::memory_order_relaxed);
    return result;
}

bool PostBox::empty() const noexcept {
    std::lock_guard lock(mutex_);
    return chunks_.empty();
}

std::size_t PostBox::spill(
    BufferResource* br,
    Communicator::Logger& log,
    rmm::cuda_stream_view stream,
    std::size_t amount
) {
    std::lock_guard lock(mutex_);
    auto max_spillable = std::transform_reduce(
        chunks_.begin(), chunks_.end(), std::size_t{0}, std::plus{}, [](auto&& chunk) {
            return chunk->data_size();
        }
    );
    auto do_spill = [&](auto&& indices) {
        for (std::size_t i : indices) {
            auto&& chunk = chunks_[i];
            auto [reservation, overbooking] =
                br->reserve(MemoryType::HOST, chunk->data_size(), true);
            if (overbooking) {
                log.warn(
                    "Cannot spill to host because of host memory overbooking: ",
                    format_nbytes(overbooking)
                );
                continue;
            }
            chunk->attach_data_buffer(
                br->move(chunk->release_data_buffer(), stream, reservation)
            );
        };
    };
    if (max_spillable < amount) {
        // need to spill everything.
        do_spill(std::views::iota(std::size_t{0}, chunks_.size()));
        return max_spillable;
    }
    auto iota = std::views::iota(std::size_t{0}, chunks_.size());
    std::vector<std::size_t> device_chunks(iota.begin(), iota.end());
    std::vector<std::size_t> to_spill;
    std::ranges::sort(device_chunks, std::less{}, [&](std::size_t i) {
        return chunks_[i]->data_size();
    });
    std::size_t total_spilled{0};
    // Try and spill the minimum number of buffers summing to the
    // amount we need while minimising the amount of data we need to
    // spill.
    while (true) {
        auto pos = std::ranges::lower_bound(
            device_chunks, amount, std::less{}, [&](std::size_t i) {
                return chunks_[i]->data_size();
            }
        );
        auto found = pos == device_chunks.end() ? device_chunks.back() : *pos;
        to_spill.push_back(found);
        if ((total_spilled += chunks_[found]->data_size()) >= amount) {
            break;
        }
        device_chunks.pop_back();
    }
    return total_spilled;
}

static std::vector<std::unique_ptr<Chunk>> test_some(
    std::vector<std::pair<std::unique_ptr<Chunk>, std::unique_ptr<Communicator::Future>>>&
        chunks,
    Communicator* comm
) {
    if (chunks.empty()) {
        return {};
    }
    std::vector<std::unique_ptr<Chunk>> result;
    std::vector<std::unique_ptr<Communicator::Future>> futures;
    futures.reserve(chunks.size());
    std::ranges::transform(chunks, std::back_inserter(futures), [](auto&& c) {
        return std::move(c.second);
    });
    auto complete = comm->test_some(futures);
    std::ranges::transform(
        complete, chunks, std::back_inserter(result), [&](auto&& fut, auto&& c) {
            auto chunk = std::move(c.first);
            RAPIDSMPF_EXPECTS(c.second == nullptr, "Oh no");
            auto data = comm->get_gpu_data(std::move(fut));
            chunk->attach_data_buffer(std::move(data));
            return std::move(chunk);
        }
    );
    auto cit = chunks.begin() + static_cast<std::int64_t>(complete.size());
    auto fit = futures.begin();
    for (; cit != chunks.end() && fit != futures.end(); cit++, fit++) {
        RAPIDSMPF_EXPECTS(*fit, "Not expecting nullptr here");
        RAPIDSMPF_EXPECTS(!(*cit).second, "Expecting no future here");
        std::swap((*cit).second, *fit);
    }
    auto osize = chunks.size();
    std::erase(
        chunks,
        std::pair<std::unique_ptr<Chunk>, std::unique_ptr<Communicator::Future>>{
            nullptr, nullptr
        }
    );
    RAPIDSMPF_EXPECTS(chunks.size() == osize - result.size(), "Something went wrong");
    return result;
}
}  // namespace detail

class AllGather::Progress {
  public:
    Progress(AllGather& gather) : gather_{gather} {}

    ProgressThread::ProgressState operator()() {
        return gather_.event_loop();
    }

  private:
    AllGather& gather_;
};

void AllGather::insert(PackedData&& packed_data) {
    if (packed_data.data->size == 0) {
        // No point communicating zero-sized insertions.
        // Note: this means the caller must handle the metadata
        // aspects of the case where the extraction from an allgather
        // is empty.
        return;
    }
    nlocal_insertions_.fetch_add(1, std::memory_order_relaxed);
    return insert(
        detail::Chunk::from_packed_data(
            sequence_number_.fetch_add(1, std::memory_order_relaxed),
            comm_->rank(),
            std::move(packed_data)
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
            static_cast<std::uint32_t>(
                nlocal_insertions_.load(std::memory_order_acquire)
            ),
            comm_->rank()
        )
    );
}

bool AllGather::finished() const noexcept {
    return finish_counter_.load(std::memory_order_acquire) == 0
           && for_extraction_.ready();
}

std::vector<PackedData> AllGather::wait_and_extract(AllGather::Ordered ordered) {
    wait();
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

void AllGather::wait() {
    can_extract_.wait(false, std::memory_order_acquire);
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
        spilled = for_extraction_.spill(br_, comm_->logger(), stream_, spill_need);
        if (spilled < spill_need) {
            spilled +=
                inserted_.spill(br_, comm_->logger(), stream_, spill_need - spilled);
        }
    }
    return spilled;
}

AllGather::~AllGather() {
    if (active_.load(std::memory_order_acquire)) {
        active_.store(false, std::memory_order_release);
        progress_thread_->remove_function(function_id_);
    }
}

AllGather::AllGather(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    OpID op_id,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
)
    : comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      stream_{stream},
      br_{br},
      statistics_{std::move(statistics)},
      finish_counter_{comm_->nranks()},
      op_id_{op_id} {
    function_id_ =
        progress_thread_->add_function([progress = std::make_shared<Progress>(*this)]() {
            return (*progress)();
        });
    spill_id_ = br_->spill_manager().add_spill_function(
        [this](std::size_t amount) -> std::size_t { return spill(amount); },
        /* priority = */ 0
    );
}

ProgressThread::ProgressState AllGather::event_loop() {
    // Data flow:
    // User inserts into inserted_
    // Send side: inserted -> extract ready -> send metadata to dst
    // and post send for buffer -> move to outbox (ready for user)
    // Receive side: receive metadata from src -> allocate chunk -> post receive
    // from src -> if origin == dst move to outbox else put in inserted
    // Note: we commit at the point of sending metadata of ready
    // messages to send data immediately. This avoids the need for one
    // ack round-trip.
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
                finish_counter_.fetch_sub(1, std::memory_order_relaxed);
                for_extraction_.increment_goalpost(chunk->sequence());
            } else {
                RAPIDSMPF_EXPECTS(
                    chunk->data_size() > 0, "Not expecting zero-sized data chunks"
                );
                for_extraction_.insert(std::move(chunk));
            }
        }
    } else {
        // Chunks that are ready to send
        for (auto&& chunk : inserted_.extract_ready()) {
            // Tell the destination about them.
            fire_and_forget_.push_back(
                comm_->send(chunk->serialize(), dst, metadata_tag, br_)
            );
            if (chunk->is_finish()) {
                finish_counter_.fetch_sub(1, std::memory_order_relaxed);
                // Finish chunk contains as sequence number the
                // number of insertions from that rank.
                for_extraction_.increment_goalpost(chunk->sequence());
            } else {
                RAPIDSMPF_EXPECTS(
                    chunk->data_size() > 0, "Not expecting zero-sized data chunks"
                );
                auto buf = chunk->release_data_buffer();
                sent_.emplace_back(
                    std::move(chunk), comm_->send(std::move(buf), dst, gpu_data_tag)
                );
            }
        }
        while (true) {
            auto const msg = comm_->recv_from(src, metadata_tag);
            if (!msg) {
                break;
            }
            auto chunk = detail::Chunk::deserialize(*msg, stream_, br_);
            if (chunk->is_finish()) {
                if (chunk->origin() != dst) {
                    // Finish chunk, if we're not the end of the ring, must forward on.
                    // We will notice this finish when we extract this chunk.
                    inserted_.insert(std::move(chunk));
                } else {
                    // Otherwise, record we're done with data from that rank.
                    finish_counter_.fetch_sub(1, std::memory_order_relaxed);
                    for_extraction_.increment_goalpost(chunk->sequence());
                }
            } else {
                RAPIDSMPF_EXPECTS(
                    chunk->data_size() > 0, "Not expecting zero-sized data chunks"
                );
                RAPIDSMPF_EXPECTS(chunk->metadata_size() > 0, "Data without metadata?!");
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
            received_.emplace_back(
                std::move(chunk), comm_->recv(src, gpu_data_tag, std::move(buf))
            );
        }
        std::erase(to_receive_, nullptr);

        std::ranges::for_each(
            detail::test_some(received_, comm_.get()), [&](auto&& chunk) {
                if (chunk->origin() == dst) {
                    for_extraction_.insert(std::move(chunk));
                } else {
                    inserted_.insert(std::move(chunk));
                }
            }
        );
        for_extraction_.insert(detail::test_some(sent_, comm_.get()));
        if (!fire_and_forget_.empty()) {
            std::ignore = comm_->test_some(fire_and_forget_);
        }
    }
    bool const containers_empty =
        (fire_and_forget_.empty() && sent_.empty() && received_.empty()
         && to_receive_.empty() && inserted_.empty());
    bool const is_finished = finished();
    bool const is_done =
        !active_.load(std::memory_order_acquire) || (is_finished && containers_empty);
    if (is_finished) {
        // We can release our output buffers so notify a waiter.
        can_extract_.store(true, std::memory_order_release);
        can_extract_.notify_one();
    }
    return is_done ? ProgressThread::ProgressState::Done
                   : ProgressThread::ProgressState::InProgress;
}

}  // namespace rapidsmpf::allgather
