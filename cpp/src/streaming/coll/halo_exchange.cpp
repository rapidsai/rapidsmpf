/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/streaming/coll/halo_exchange.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

namespace rapidsmpf::streaming {

namespace {

// ---------------------------------------------------------------------------
// Wire format for one direction's metadata message:
//
//   [uint64_t gpu_data_size (8 bytes)] [uint8_t... cudf_metadata_bytes]
//
// Null sentinel: exactly 8 bytes with gpu_data_size == 0 means the sender
// has nothing to send (boundary rank or explicit nullopt).  This lets every
// participating rank always send exactly one metadata message per direction
// per round, keeping the recv_from polling simple.
// ---------------------------------------------------------------------------

std::unique_ptr<std::vector<std::uint8_t>> make_meta_msg(
    std::optional<PackedData> const& pd
) {
    if (!pd.has_value()) {
        // Null sentinel
        auto msg = std::make_unique<std::vector<std::uint8_t>>(sizeof(std::uint64_t));
        std::uint64_t const zero = 0;
        std::memcpy(msg->data(), &zero, sizeof(zero));
        return msg;
    }
    auto const& meta = *pd->metadata;
    std::uint64_t const dsize = pd->data->size;
    auto msg =
        std::make_unique<std::vector<std::uint8_t>>(sizeof(std::uint64_t) + meta.size());
    std::memcpy(msg->data(), &dsize, sizeof(dsize));
    std::memcpy(msg->data() + sizeof(std::uint64_t), meta.data(), meta.size());
    return msg;
}

bool is_null_msg(std::vector<std::uint8_t> const& msg) {
    if (msg.size() != sizeof(std::uint64_t)) {
        return false;
    }
    std::uint64_t dsize{};
    std::memcpy(&dsize, msg.data(), sizeof(dsize));
    return dsize == 0;
}

std::uint64_t parse_data_size(std::vector<std::uint8_t> const& msg) {
    std::uint64_t dsize{};
    std::memcpy(&dsize, msg.data(), sizeof(dsize));
    return dsize;
}

std::unique_ptr<std::vector<std::uint8_t>> parse_metadata(
    std::vector<std::uint8_t> const& msg
) {
    auto meta =
        std::make_unique<std::vector<std::uint8_t>>(msg.size() - sizeof(std::uint64_t));
    std::memcpy(meta->data(), msg.data() + sizeof(std::uint64_t), meta->size());
    return meta;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

HaloExchange::HaloExchange(
    std::shared_ptr<Context> ctx, std::shared_ptr<Communicator> comm, OpID op_id
)
    : ctx_{std::move(ctx)}, comm_{std::move(comm)}, op_id_{op_id} {
    function_id_ =
        comm_->progress_thread()->add_function([this]() { return event_loop(); });
}

HaloExchange::~HaloExchange() noexcept {
    RAPIDSMPF_EXPECTS_FATAL(
        event_.is_set(),
        "~HaloExchange: destroyed while an exchange() round is still pending; "
        "the caller must await exchange() to completion before destroying this object"
    );
    active_.store(false, std::memory_order_release);
    comm_->progress_thread()->remove_function(function_id_);
}

// ---------------------------------------------------------------------------
// exchange()
// ---------------------------------------------------------------------------

coro::task<std::pair<std::optional<PackedData>, std::optional<PackedData>>>
HaloExchange::exchange(
    std::optional<PackedData> send_left, std::optional<PackedData> send_right
) {
    Rank const rank = comm_->rank();
    Rank const nranks = comm_->nranks();

    {
        std::lock_guard lock(mutex_);

        // Reset per-round receive state
        left_done_ = (rank == 0);  // no left neighbor → already done
        right_done_ = (rank == nranks - 1);  // no right neighbor → already done
        left_meta_received_ = false;
        right_meta_received_ = false;
        from_left_ = std::nullopt;
        from_right_ = std::nullopt;
        left_data_recv_future_ = nullptr;
        right_data_recv_future_ = nullptr;
        left_metadata_ = nullptr;
        right_metadata_ = nullptr;
        left_data_size_ = 0;
        right_data_size_ = 0;

        event_.reset();

        // --- Issue sends ---
        // Rightward: this rank → rank+1   (stages 0=meta, 1=GPU data)
        if (rank < nranks - 1) {
            sends_.push_back(
                comm_->send(make_meta_msg(send_right), rank + 1, Tag{op_id_, 0})
            );
            if (send_right.has_value()) {
                sends_.push_back(
                    comm_->send(std::move(send_right->data), rank + 1, Tag{op_id_, 1})
                );
            }
        }
        // Leftward: this rank → rank-1   (stages 2=meta, 3=GPU data)
        if (rank > 0) {
            sends_.push_back(
                comm_->send(make_meta_msg(send_left), rank - 1, Tag{op_id_, 2})
            );
            if (send_left.has_value()) {
                sends_.push_back(
                    comm_->send(std::move(send_left->data), rank - 1, Tag{op_id_, 3})
                );
            }
        }

        // Single-rank fast path: nothing to receive, signal immediately
        if (left_done_ && right_done_) {
            event_.set();
        }
    }  // release mutex before suspending

    co_await event_;

    std::lock_guard lock(mutex_);
    co_return {std::move(from_left_), std::move(from_right_)};
}

// ---------------------------------------------------------------------------
// event_loop()  — called repeatedly by the progress thread
// ---------------------------------------------------------------------------

ProgressThread::ProgressState HaloExchange::event_loop() {
    RAPIDSMPF_NVTX_SCOPED_RANGE_VERBOSE("HaloExchange::event_loop");
    std::lock_guard lock(mutex_);

    // Retire completed send futures
    if (!sends_.empty()) {
        std::ignore = comm_->test_some(sends_);
    }

    Rank const rank = comm_->rank();
    Rank const nranks = comm_->nranks();

    // -----------------------------------------------------------------------
    // Receive from LEFT neighbor (rank-1 sent rightward → stages 0=meta, 1=data)
    // -----------------------------------------------------------------------
    if (rank > 0 && !left_done_) {
        if (!left_meta_received_) {
            auto msg = comm_->recv_from(rank - 1, Tag{op_id_, 0});
            if (msg) {
                if (is_null_msg(*msg)) {
                    left_done_ = true;
                } else {
                    left_data_size_ = parse_data_size(*msg);
                    left_metadata_ = parse_metadata(*msg);
                    if (left_data_size_ == 0) {
                        // Metadata only (e.g. 0-row table): no GPU recv needed
                        from_left_ = PackedData{
                            std::move(left_metadata_),
                            ctx_->br()->allocate(
                                ctx_->br()->stream_pool().get_stream(),
                                ctx_->br()->reserve_or_fail(0, MEMORY_TYPES)
                            )
                        };
                        left_done_ = true;
                    } else {
                        // Allocate device buffer and post async GPU receive
                        auto buf = ctx_->br()->allocate(
                            ctx_->br()->stream_pool().get_stream(),
                            ctx_->br()->reserve_or_fail(left_data_size_, MEMORY_TYPES)
                        );
                        left_data_recv_future_ =
                            comm_->recv(rank - 1, Tag{op_id_, 1}, std::move(buf));
                        left_meta_received_ = true;  // means: GPU recv posted for stage 1
                    }
                }
            }
        }
        if (left_meta_received_ && left_data_recv_future_) {
            if (comm_->test(left_data_recv_future_)) {
                from_left_ = PackedData{
                    std::move(left_metadata_),
                    comm_->release_data(std::move(left_data_recv_future_))
                };
                left_done_ = true;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Receive from RIGHT neighbor (rank+1 sent leftward → stages 2=meta, 3=data)
    // -----------------------------------------------------------------------
    if (rank < nranks - 1 && !right_done_) {
        if (!right_meta_received_) {
            auto msg = comm_->recv_from(rank + 1, Tag{op_id_, 2});
            if (msg) {
                if (is_null_msg(*msg)) {
                    right_done_ = true;
                } else {
                    right_data_size_ = parse_data_size(*msg);
                    right_metadata_ = parse_metadata(*msg);
                    if (right_data_size_ == 0) {
                        from_right_ = PackedData{
                            std::move(right_metadata_),
                            ctx_->br()->allocate(
                                ctx_->br()->stream_pool().get_stream(),
                                ctx_->br()->reserve_or_fail(0, MEMORY_TYPES)
                            )
                        };
                        right_done_ = true;
                    } else {
                        auto buf = ctx_->br()->allocate(
                            ctx_->br()->stream_pool().get_stream(),
                            ctx_->br()->reserve_or_fail(right_data_size_, MEMORY_TYPES)
                        );
                        right_data_recv_future_ =
                            comm_->recv(rank + 1, Tag{op_id_, 3}, std::move(buf));
                        right_meta_received_ =
                            true;  // means: GPU recv posted for stage 3
                    }
                }
            }
        }
        if (right_meta_received_ && right_data_recv_future_) {
            if (comm_->test(right_data_recv_future_)) {
                from_right_ = PackedData{
                    std::move(right_metadata_),
                    comm_->release_data(std::move(right_data_recv_future_))
                };
                right_done_ = true;
            }
        }
    }

    // Wake the coroutine once both directions are resolved.
    // event_.set(executor) schedules resumption on the executor rather than
    // resuming the waiting coroutine inline.  Scheduled continuations therefore
    // never run from inside set(), so holding mutex_ here is safe: the resumed
    // coroutine will acquire mutex_ only after this lock_guard is released.
    if (left_done_ && right_done_ && !event_.is_set()) {
        event_.set(ctx_->executor()->get());
    }

    bool const all_clean =
        sends_.empty() && !left_data_recv_future_ && !right_data_recv_future_;
    return (!active_.load(std::memory_order_acquire) && all_clean)
               ? ProgressThread::ProgressState::Done
               : ProgressThread::ProgressState::InProgress;
}

}  // namespace rapidsmpf::streaming
