/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <rapidsmpf/coll/allreduce.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/progress_thread.hpp>

namespace rapidsmpf::coll {

AllReduce::AllReduce(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ProgressThread> progress_thread,
    std::unique_ptr<Buffer> input,
    std::unique_ptr<Buffer> output,
    OpID op_id,
    ReduceOperator reduce_operator,
    std::function<void(void)> finished_callback
)
    : comm_{std::move(comm)},
      progress_thread_{std::move(progress_thread)},
      reduce_operator_{std::move(reduce_operator)},
      in_buffer_{std::move(input)},
      out_buffer_{std::move(output)},
      op_id_{op_id},
      finished_callback_{std::move(finished_callback)},
      nearest_pow2_{
          static_cast<Rank>(std::bit_floor(static_cast<std::uint32_t>(comm_->nranks())))
      },
      non_pow2_remainder_{comm_->nranks() - nearest_pow2_} {
    RAPIDSMPF_EXPECTS(
        in_buffer_ != nullptr,
        "AllReduce requires a valid input buffer",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        out_buffer_ != nullptr,
        "AllReduce requires a valid output buffer",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        in_buffer_->size == out_buffer_->size,
        "AllReduce requires input/output buffer sizes to match",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        in_buffer_->mem_type() == out_buffer_->mem_type(),
        "AllReduce requires input/output buffer memory types to match",
        std::invalid_argument

    );
    out_buffer_->rebind_stream(in_buffer_->stream());
    // Note: after this copy, we must check out_buffer's write event before receiving into
    // in_buffer. See StartPreRemainder in the event loop.
    buffer_copy(*out_buffer_, *in_buffer_, in_buffer_->size);

    auto const rank = comm_->rank();
    if (rank < 2 * non_pow2_remainder_) {
        if (rank % 2 == 0) {
            logical_rank_ = -1;
        } else {
            logical_rank_ = rank / 2;
        }
        phase_ = Phase::StartPreRemainder;
    } else {
        logical_rank_ = rank - non_pow2_remainder_;
        phase_ = Phase::StartButterfly;
    }
    function_id_ = progress_thread_->add_function([this]() { return event_loop(); });
}

AllReduce::~AllReduce() noexcept {
    if (function_id_.is_valid() && progress_thread_
        && active_.load(std::memory_order_acquire))
    {
        auto const phase = phase_.load(std::memory_order_acquire);
        if (phase != Phase::ResultAvailable && phase != Phase::Done) {
            // If we get here and we hadn't finished the event loop then there are in
            // flight messages that will be lost forever and potentially be matched
            // incorrectly so there's really nothing we can do.
            comm_->logger().warn("Destroying AllReduce before waiting for extraction");
            std::terminate();
        }
        active_.store(false, std::memory_order_release);
        progress_thread_->remove_function(function_id_);
    }
}

bool AllReduce::finished() const noexcept {
    return phase_.load(std::memory_order_acquire) == Phase::ResultAvailable;
}

std::pair<std::unique_ptr<Buffer>, std::unique_ptr<Buffer>> AllReduce::wait_and_extract(
    std::chrono::milliseconds timeout
) {
    std::unique_lock lock(mutex_);

    if (timeout.count() < 0) {
        cv_.wait(lock, [this] {
            return phase_.load(std::memory_order_acquire) == Phase::ResultAvailable;
        });
    } else {
        bool completed = cv_.wait_for(lock, timeout, [this] {
            return phase_.load(std::memory_order_acquire) == Phase::ResultAvailable;
        });
        if (!completed) {
            RAPIDSMPF_FAIL(
                "AllReduce::wait_and_extract timed out waiting for reduction to "
                "complete",
                std::runtime_error
            );
        }
    }

    RAPIDSMPF_EXPECTS(
        in_buffer_ != nullptr && out_buffer_ != nullptr,
        "AllReduce::wait_and_extract can only be called once",
        std::runtime_error
    );
    return {std::move(in_buffer_), std::move(out_buffer_)};
}

bool AllReduce::is_ready() const noexcept {
    return phase_.load(std::memory_order_acquire) == Phase::ResultAvailable;
}

ProgressThread::ProgressState AllReduce::event_loop() {
    Rank const rank = comm_->rank();
    bool const is_even = rank % 2 == 0;
    // We only need a single stage ID because of no-message-overtaking guarantees in the
    // communicator. We could use multiple stage IDs for each round of the exchange, but
    // that would break once we have more than 256 participating ranks: there are only
    // three bits available for the stage in the tag and we have log_2 nranks rounds.
    // In any case, only needing a single tag is nice.
    Tag tag{op_id_, 0};
    if (!active_.load(std::memory_order_acquire)) {
        return ProgressThread::ProgressState::Done;
    }
    // Communication pattern. If the ranks form a power of two, then they exchange
    // messages and combine via recursive doubling:
    //
    //      +-----------finished doubling------+
    //      |                                  |
    //      |                                  v
    // StartButterfly -> CompleteButterfly -> Done
    //      ^                   |
    //      |                   |
    //      +-----doubling------+
    //
    // If we don't have a power of two number of ranks, then we have `power_of_two +
    // remainder` ranks. We first take `2 * remainder` ranks, and the even ranks send
    // their contribution to their odd pair. The even ranks then jump to receive a final
    // contribution, while the rest form a power of two and exchange via the above loop.
    // Once that is complete, the paired odd ranks send the final answer to their even
    // counterpart.
    //
    // So even ranks in the remainder do:
    // PreRemainder -> PostRemainder -> Done
    // odd ranks in the remainder do:
    // PreRemainder -> Butterfly -> PostRemainder -> Done
    switch (phase_.load(std::memory_order_acquire)) {
    case Phase::StartPreRemainder:
        {
            // Non-participating ranks have jumped straight to StartButterfly
            if (is_even) {
                if (!out_buffer_->is_latest_write_done()) {
                    break;
                }
                send_future_ = comm_->send(std::move(out_buffer_), rank + 1, tag);
            } else {
                // The constructor copies in_buffer_ to out_buffer_ on in_buffer's
                // stream. The copy must be complete before we can receive into
                // in_buffer_ otherwise we have a write-after-read hazard.
                // Note that buffer_ready(out_buffer_) tracks the WAR hazard, not
                // buffer_ready(in_buffer_). Checking the readiness of in_buffer_ is
                // belt-and-braces.
                if (!out_buffer_->is_latest_write_done()
                    || !in_buffer_->is_latest_write_done())
                {
                    break;
                }
                recv_future_ = comm_->recv(rank - 1, tag, std::move(in_buffer_));
            }
            phase_.store(Phase::CompletePreRemainder, std::memory_order_release);
            break;
        }
    case Phase::CompletePreRemainder:
        {
            if (is_even) {
                if (!comm_->test(send_future_)) {
                    break;
                }
                out_buffer_ = comm_->release_data(std::move(send_future_));
                // OK, we've sent our contributions to our partner rank, now we just have
                // to wait for the response once its down with the rest of the reduction.
                phase_.store(Phase::StartPostRemainder, std::memory_order_release);
            } else {
                if (!comm_->test(recv_future_)) {
                    break;
                }
                in_buffer_ = comm_->release_data(std::move(recv_future_));
                reduce_operator_(in_buffer_.get(), out_buffer_.get());
                phase_.store(Phase::StartButterfly, std::memory_order_release);
            }
            break;
        }
    case Phase::StartButterfly:
        {
            // This condition is here rather than at the end of the completion when we
            // update the mask to handle the case where there is only one rank.
            if (stage_mask_ >= nearest_pow2_) {
                if (rank < 2 * non_pow2_remainder_) {
                    phase_.store(Phase::StartPostRemainder, std::memory_order_release);
                } else {
                    phase_.store(Phase::Done, std::memory_order_release);
                }
                break;
            }
            auto const logical_dst = logical_rank_ ^ stage_mask_;
            stage_partner_ = logical_dst < non_pow2_remainder_
                                 ? logical_dst * 2 + 1
                                 : logical_dst + non_pow2_remainder_;
            // As with the copy in the ctor, the reduce_operator_ is stream-ordered on
            // out_buffer_'s stream and reads in_buffer_. So we must guard for the WAR
            // hazard before receiving in the next round.
            // That, again, is tracked by out_buffer_ being ready and in_buffer readiness
            // is belt-and-braces.
            if (!out_buffer_->is_latest_write_done()
                || !in_buffer_->is_latest_write_done())
            {
                break;
            }
            recv_future_ = comm_->recv(stage_partner_, tag, std::move(in_buffer_));
            send_future_ = comm_->send(std::move(out_buffer_), stage_partner_, tag);
            phase_.store(Phase::CompleteButterfly, std::memory_order_release);
            break;
        }
    case Phase::CompleteButterfly:
        {
            if (!comm_->test(recv_future_) || !comm_->test(send_future_)) {
                break;
            }
            in_buffer_ = comm_->release_data(std::move(recv_future_));
            out_buffer_ = comm_->release_data(std::move(send_future_));
            // Swapped operand order for the case where the operator is non-commutative.
            // This ensures everyone combines in the same order and means that for a given
            // input and given number of ranks, everyone always obtains the same result
            // even if the operator is neither associative nor commutative.
            if (stage_partner_ < rank) {
                reduce_operator_(in_buffer_.get(), out_buffer_.get());
            } else {
                reduce_operator_(out_buffer_.get(), in_buffer_.get());
                std::swap(out_buffer_, in_buffer_);
            }
            stage_mask_ <<= 1;
            phase_.store(Phase::StartButterfly, std::memory_order_release);
            break;
        }
    case Phase::StartPostRemainder:
        {
            if (is_even) {
                if (!out_buffer_->is_latest_write_done()) {
                    break;
                }
                recv_future_ = comm_->recv(rank + 1, tag, std::move(out_buffer_));
            } else {
                if (!out_buffer_->is_latest_write_done()) {
                    break;
                }
                send_future_ = comm_->send(std::move(out_buffer_), rank - 1, tag);
            }
            phase_.store(Phase::CompletePostRemainder, std::memory_order_release);
            break;
        }
    case Phase::CompletePostRemainder:
        {
            if (is_even) {
                if (!comm_->test(recv_future_)) {
                    break;
                }
                out_buffer_ = comm_->release_data(std::move(recv_future_));
            } else {
                if (!comm_->test(send_future_)) {
                    break;
                }
                out_buffer_ = comm_->release_data(std::move(send_future_));
            }
            phase_.store(Phase::Done, std::memory_order_release);
            break;
        }
    case Phase::ResultAvailable:
        RAPIDSMPF_FAIL("Event loop should never read ResultAvailable");
    case Phase::Done:
    default:
        break;
    }

    if (phase_.load(std::memory_order_acquire) == Phase::Done) {
        {
            std::lock_guard lock(mutex_);
            phase_.store(Phase::ResultAvailable, std::memory_order_release);
            if (finished_callback_) {
                finished_callback_();
            }
        }
        cv_.notify_all();
        return ProgressThread::ProgressState::Done;
    }

    return ProgressThread::ProgressState::InProgress;
}

}  // namespace rapidsmpf::coll
