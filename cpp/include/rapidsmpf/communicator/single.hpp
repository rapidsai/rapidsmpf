/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/config.hpp>

namespace rapidsmpf {

/**
 * @brief Single process communicator class that implements the `Communicator` interface.
 *
 * This class stubs out the `Communicator` interface with functions that throw.
 * When sending to/receiving from self the internal logic should move
 * buffers through the shuffler, rather than invoking send/recv.
 */
class Single final : public Communicator {
  public:
    /**
     * @brief Represents the future result of an operation.
     *
     * This class is used to handle the result of a communication operation
     * asynchronously.
     */
    class Future : public Communicator::Future {
        friend class Single;

      public:
        ~Future() noexcept override = default;
    };

    /**
     * @brief Construct a single process communicator.
     *
     * @param options Configuration options.
     */
    Single(config::Options options);

    ~Single() noexcept override = default;

    /**
     * @copydoc Communicator::rank
     */
    [[nodiscard]] constexpr Rank rank() const override {
        return 0;
    }

    /**
     * @copydoc Communicator::nranks
     */
    [[nodiscard]] constexpr Rank nranks() const override {
        return 1;
    }

    /**
     * @copydoc Communicator::send
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<std::vector<uint8_t>> msg, Rank rank, Tag tag, BufferResource* br
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::send(std::unique_ptr<Buffer> msg, Rank rank, Tag tag)
     *
     * @throws std::runtime_error if called (single-process communicators should never send messages).
     */
    // clang-format on
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, Tag tag
    ) override;

    /**
     * @copydoc Communicator::recv
     *
     * @throws std::runtime_error if called (single-process communicators should never
     * send messages).
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> recv(
        Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
    ) override;

    /**
     * @copydoc Communicator::recv_any
     *
     * @note Always returns a nullptr for the received message, indicating that no message
     * is available.
     */
    [[nodiscard]] std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(Tag tag
    ) override;

    /**
     * @copydoc Communicator::test_some
     *
     * @throws std::runtime_error if called (single-process communicators should never
     * send messages).
     */
    std::vector<std::size_t> test_some(
        std::vector<std::unique_ptr<Communicator::Future>> const& future_vector
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::test_some(std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const& future_map)
     *
     * @throws std::runtime_error if called (single-process communicators should never send messages).
     */
    // clang-format on
    std::vector<std::size_t> test_some(
        std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
            future_map
    ) override;

    /**
     * @copydoc Communicator::get_gpu_data
     *
     * @throws std::runtime_error if called (single-process communicators should never
     * send messages).
     */
    [[nodiscard]] std::unique_ptr<Buffer> get_gpu_data(
        std::unique_ptr<Communicator::Future> future
    ) override;

    /**
     * @copydoc Communicator::logger
     */
    [[nodiscard]] Logger& logger() override {
        return logger_;
    }

    /**
     * @copydoc Communicator::str
     */
    [[nodiscard]] std::string str() const override;

  private:
    Logger logger_;
};


}  // namespace rapidsmpf
