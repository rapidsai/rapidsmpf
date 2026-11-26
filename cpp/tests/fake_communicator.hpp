/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rapidsmpf/communicator/communicator.hpp>

/**
 * @brief Communicator posing as rank 0 of 2 ranks, whose communication never completes
 * or fails.
 *
 * Sends and receives only complete once `complete_futures()` is called, or, for
 * receives, once cancelled when the communicator is destroyed. No message ever arrives.
 * Depending on its `Mode`, the communication fails instead. Like the real backends, a
 * future destroyed before its communication completed hands it over to the graveyard of
 * the communicator. Used to test the handling of in-flight communication and of
 * communication failures deterministically, without communicating with any other rank.
 * The logger and progress thread are borrowed from a real communicator.
 */
class FakeCommunicator final : public rapidsmpf::Communicator {
  public:
    /// @brief The message of the exceptions thrown by the communication operations.
    static constexpr char const* error_message = "injected communicator failure";

    /// @brief How the communication behaves.
    enum class Mode : std::uint8_t {
        /// @brief Sends and receives succeed, returning futures that never complete.
        InFlight,
        /// @brief All communication throws.
        FailAll,
        /// @brief Like `InFlight`, but polling throws once communication is in flight,
        /// so that operations fail with communication in flight.
        FailWithInFlight,
    };

    /// @brief The direction of a communication operation.
    enum class Direction : std::uint8_t {
        Send,  ///< A send, never cancelled.
        Receive,  ///< A receive, cancelled when the communicator is destroyed.
    };

    /// @brief The state shared by the communicator and its communication.
    struct State {
        std::atomic<bool> completed{false};  ///< Whether all communication completed.
        std::atomic<int> live{0};  ///< Communication not freed, e.g., leaked.
        /// @brief Communication freed while in flight: a use-after-free with a real
        /// communication library.
        std::atomic<int> freed_in_flight{0};
    };

    /// @brief A posted send or receive.
    class Communication {
      public:
        Communication(std::shared_ptr<State> state, Direction direction)
            : state_{std::move(state)}, direction_{direction} {
            ++state_->live;
        }

        ~Communication() noexcept {
            if (!completed()) {
                ++state_->freed_in_flight;
            }
            --state_->live;
        }

        Communication(Communication const&) = delete;
        Communication& operator=(Communication const&) = delete;

        /// @return Whether the communication completed.
        [[nodiscard]] bool completed() const noexcept {
            return state_->completed.load() || cancelled_;
        }

        /// @brief Cancel the communication, which completes receives right away.
        void cancel() noexcept {
            cancelled_ = direction_ == Direction::Receive;
        }

      private:
        std::shared_ptr<State> state_;
        Direction direction_;
        bool cancelled_{false};
    };

    /// @brief The communication of a destroyed future, see `FutureGraveyard`.
    class Grave final : public FutureGraveyard::Grave {
      public:
        explicit Grave(std::unique_ptr<Communication> communication)
            : communication_{std::move(communication)} {}

        [[nodiscard]] bool completed() noexcept override {
            return communication_->completed();
        }

        void cancel() noexcept override {
            communication_->cancel();
        }

      private:
        std::unique_ptr<Communication> communication_;
    };

    /// @brief The future of a posted send or receive.
    class PendingFuture final : public Future {
      public:
        PendingFuture(
            std::unique_ptr<Communication> communication,
            std::shared_ptr<FutureGraveyard> graveyard
        )
            : communication_{std::move(communication)},
              graveyard_{std::move(graveyard)} {}

        ~PendingFuture() noexcept override {
            if (!communication_->completed()) {
                graveyard_->bury(std::make_unique<Grave>(std::move(communication_)));
            }
        }

        PendingFuture(PendingFuture&&) = delete;
        PendingFuture& operator=(PendingFuture&&) = delete;

      private:
        std::unique_ptr<Communication> communication_;
        std::shared_ptr<FutureGraveyard> graveyard_;
    };

    explicit FakeCommunicator(
        std::shared_ptr<rapidsmpf::Communicator> comm, Mode mode = Mode::InFlight
    )
        : comm_{std::move(comm)}, mode_{mode} {}

    ~FakeCommunicator() noexcept override {
        graveyard()->release();
    }

    /// @brief Make the communication of this communicator complete.
    void complete_futures() {
        state_->completed = true;
    }

    /// @return The state shared with the communication, which outlives the communicator.
    [[nodiscard]] std::shared_ptr<State> const& state() const {
        return state_;
    }

    [[nodiscard]] rapidsmpf::Rank rank() const override {
        return 0;
    }

    [[nodiscard]] rapidsmpf::Rank nranks() const override {
        return 2;
    }

    [[nodiscard]] std::unique_ptr<Future> send(
        std::unique_ptr<std::vector<std::uint8_t>>, rapidsmpf::Rank, rapidsmpf::Tag
    ) override {
        return post(Direction::Send);
    }

    [[nodiscard]] std::unique_ptr<Future> send(
        std::unique_ptr<rapidsmpf::Buffer>, rapidsmpf::Rank, rapidsmpf::Tag
    ) override {
        return post(Direction::Send);
    }

    [[nodiscard]] std::unique_ptr<Future> recv(
        rapidsmpf::Rank, rapidsmpf::Tag, std::unique_ptr<rapidsmpf::Buffer>
    ) override {
        return post(Direction::Receive);
    }

    [[nodiscard]] std::unique_ptr<Future> recv_sync_host_data(
        rapidsmpf::Rank, rapidsmpf::Tag, std::unique_ptr<std::vector<std::uint8_t>>
    ) override {
        return post(Direction::Receive);
    }

    [[nodiscard]] std::pair<std::unique_ptr<std::vector<std::uint8_t>>, rapidsmpf::Rank>
    recv_any(rapidsmpf::Tag) override {
        throw_if_aborted();
        poll();
        return {nullptr, 0};
    }

    [[nodiscard]] std::unique_ptr<std::vector<std::uint8_t>> recv_from(
        rapidsmpf::Rank, rapidsmpf::Tag
    ) override {
        throw_if_aborted();
        poll();
        return nullptr;
    }

    [[nodiscard]] std::
        pair<std::vector<std::unique_ptr<Future>>, std::vector<std::size_t>>
        test_some(std::vector<std::unique_ptr<Future>>&) override {
        poll();
        return {};
    }

    [[nodiscard]] std::unordered_map<std::size_t, std::unique_ptr<Future>> test_some(
        std::unordered_map<std::size_t, std::unique_ptr<Future>>&
    ) override {
        poll();
        return {};
    }

    [[nodiscard]] bool test(std::unique_ptr<Future>&) override {
        poll();
        return false;
    }

    [[nodiscard]] std::vector<std::unique_ptr<rapidsmpf::Buffer>> wait_all(
        std::vector<std::unique_ptr<Future>>&&
    ) override {
        throw std::logic_error("FakeCommunicator: communication never completes");
    }

    [[nodiscard]] std::unique_ptr<rapidsmpf::Buffer> wait(
        std::unique_ptr<Future>
    ) override {
        throw std::logic_error("FakeCommunicator: communication never completes");
    }

    [[nodiscard]] std::unique_ptr<rapidsmpf::Buffer> release_data(
        std::unique_ptr<Future>
    ) override {
        throw std::logic_error("FakeCommunicator: communication never completes");
    }

    [[nodiscard]] std::unique_ptr<std::vector<std::uint8_t>> release_sync_host_data(
        std::unique_ptr<Future>
    ) override {
        throw std::logic_error("FakeCommunicator: communication never completes");
    }

    [[nodiscard]] std::shared_ptr<rapidsmpf::Logger> const& logger() override {
        return comm_->logger();
    }

    [[nodiscard]] std::shared_ptr<rapidsmpf::ProgressThread> const&
    progress_thread() const override {
        return comm_->progress_thread();
    }

    [[nodiscard]] std::string str() const override {
        return "FakeCommunicator(rank=0, nranks=2)";
    }

  private:
    /// @brief Post a send or receive.
    std::unique_ptr<Future> post(Direction direction) {
        throw_if_aborted();
        if (mode_ == Mode::FailAll) {
            throw std::runtime_error(error_message);
        }
        return std::make_unique<PendingFuture>(
            std::make_unique<Communication>(state_, direction), graveyard()
        );
    }

    /// @brief Poll for completed communication or incoming messages.
    void poll() {
        if (mode_ == Mode::FailAll
            || (mode_ == Mode::FailWithInFlight && state_->live.load() > 0))
        {
            throw std::runtime_error(error_message);
        }
    }

    std::shared_ptr<rapidsmpf::Communicator> comm_;
    Mode mode_;
    std::shared_ptr<State> state_{std::make_shared<State>()};
};

/**
 * @brief Matcher for a callable throwing the error injected by `FakeCommunicator`.
 *
 * @return The matcher.
 */
inline auto ThrowsInjectedFailure() {
    return ::testing::ThrowsMessage<std::runtime_error>(
        ::testing::StrEq(FakeCommunicator::error_message)
    );
}

/**
 * @brief Wait until `predicate` is true, giving up after `timeout`.
 *
 * @param predicate The predicate to wait for.
 * @param timeout The maximum duration to wait.
 * @return The final value of `predicate`.
 */
template <typename Predicate>
bool eventually(
    Predicate&& predicate, std::chrono::seconds timeout = std::chrono::seconds{30}
) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}
