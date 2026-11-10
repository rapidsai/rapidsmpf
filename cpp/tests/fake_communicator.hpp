/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>

/**
 * @brief Communicator posing as rank 0 of 2 ranks, whose communication never completes.
 *
 * Sends and receives return futures that only complete once cancelled, which, like
 * with the real backends, only receives are, or once `complete_futures()` is called.
 * No message ever arrives. Used to test the handling of in-flight communication
 * deterministically, without communicating with any other rank. The logger and
 * progress thread are borrowed from a real communicator.
 */
class FakeCommunicator final : public rapidsmpf::Communicator {
  public:
    /// @brief The state shared by the communicator and its futures.
    struct State {
        std::atomic<bool> completed{false};  ///< Whether all futures completed.
        std::atomic<int> dropped{0};  ///< Futures destroyed before they completed.
        std::mutex mutex;  ///< Protects `live`.
        std::unordered_set<Future*> live;  ///< The live futures.

        /// @return The number of live futures.
        [[nodiscard]] std::size_t num_live() {
            std::lock_guard lock(mutex);
            return live.size();
        }

        /// @brief Destroy the live futures, e.g., those leaked by the communicator.
        void destroy_live() {
            completed = true;
            std::unordered_set<Future*> futures;
            {
                std::lock_guard lock(mutex);
                futures = live;
            }
            for (auto* future : futures) {
                delete future;
            }
        }
    };

    /// @brief A future that only completes once cancelled or completed.
    class PendingFuture final : public Future {
      public:
        PendingFuture(std::shared_ptr<State> state, Direction direction)
            : state_{std::move(state)}, direction_{direction} {
            std::lock_guard lock(state_->mutex);
            state_->live.insert(this);
        }

        ~PendingFuture() noexcept override {
            if (!completed()) {
                ++state_->dropped;
            }
            std::lock_guard lock(state_->mutex);
            state_->live.erase(this);
        }

        /// @brief Request the cancellation, which completes receives right away.
        void cancel() noexcept {
            cancelled_ = direction_ == Direction::Receive;
        }

        /// @return Whether the communication completed.
        [[nodiscard]] bool completed() const noexcept {
            return state_->completed.load() || cancelled_;
        }

      private:
        std::shared_ptr<State> state_;
        Direction direction_;
        bool cancelled_{false};
    };

    explicit FakeCommunicator(std::shared_ptr<rapidsmpf::Communicator> comm)
        : comm_{std::move(comm)} {}

    ~FakeCommunicator() noexcept override {
        release_retired_futures();
    }

    /// @brief Make the futures returned by this communicator complete.
    void complete_futures() {
        state_->completed = true;
    }

    /// @return The state shared with the futures, which outlives the communicator.
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
        poll();
        return {nullptr, 0};
    }

    [[nodiscard]] std::unique_ptr<std::vector<std::uint8_t>> recv_from(
        rapidsmpf::Rank, rapidsmpf::Tag
    ) override {
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

  protected:
    [[nodiscard]] bool retired_future_completed(Future& future) noexcept override {
        return static_cast<PendingFuture&>(future).completed();
    }

    void cancel_retired_future(Future& future) noexcept override {
        static_cast<PendingFuture&>(future).cancel();
    }

  private:
    /// @brief Post a send or receive.
    std::unique_ptr<Future> post(Direction direction) {
        return std::make_unique<PendingFuture>(state_, direction);
    }

    /// @brief Poll for completed communication or incoming messages.
    void poll() {}

    std::shared_ptr<rapidsmpf::Communicator> comm_;
    std::shared_ptr<State> state_{std::make_shared<State>()};
};

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
