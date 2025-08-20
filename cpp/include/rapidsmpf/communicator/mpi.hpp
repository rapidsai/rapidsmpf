/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

#include <mpi.h>

#include <rmm/device_buffer.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

/**
 * @namespace rapidsmpf::mpi
 * @brief Collection of helpful [MPI](https://www.mpi-forum.org/docs/) functions.
 */
namespace mpi {

/**
 * @brief Helper to initialize MPI with threading support.
 *
 * @param argc Pointer to the number of arguments passed to the program.
 * @param argv Pointer to the argument vector passed to the program.
 */
void init(int* argc, char*** argv);

/**
 * @brief Check if MPI is initialized.
 *
 * @return true If MPI is initialized.
 */
bool is_initialized();

/**
 * @brief Helper to check the MPI errcode of an MPI call.
 *
 * A macro to check the result of an MPI call and handle any error codes that are
 * returned.
 *
 * @param call The MPI call to be checked for errors.
 */
#define RAPIDSMPF_MPI(call) \
    rapidsmpf::mpi::detail::check_mpi_error((call), __FILE__, __LINE__)

namespace detail {
/**
 * @brief Checks and reports MPI error codes.
 *
 * @param error_code The MPI error code to check.
 * @param file The file where the MPI call occurred.
 * @param line The line number where the MPI call occurred.
 */
void check_mpi_error(int error_code, const char* file, int line);
}  // namespace detail
}  // namespace mpi

/**
 * @brief MPI communicator class that implements the `Communicator` interface.
 *
 * This class implements communication functions using MPI, allowing for data exchange
 * between processes in a distributed system. It supports sending and receiving data, both
 * on the CPU and GPU, and provides asynchronous operations with support for future
 * results.
 */
class MPI final : public Communicator {
  public:
    /**
     * @brief Represents the future result of an MPI operation.
     *
     * This class is used to handle the result of an MPI communication operation
     * asynchronously.
     */
    class Future : public Communicator::Future {
        friend class MPI;

      public:
        /**
         * @brief Construct a Future.
         *
         * @param req The MPI request handle for the operation.
         * @param data A unique pointer to the data buffer.
         */
        Future(MPI_Request req, std::unique_ptr<Buffer> data)
            : req_{req}, data_{std::move(data)} {}

        ~Future() noexcept override = default;

      private:
        using Base = Communicator::Future;

        MPI_Request req_;  ///< The MPI request associated with the operation.
        std::unique_ptr<Buffer> data_;  ///< The data buffer.
    };

    /**
     * @brief Represents the future result of an MPI operation.
     *
     * This class is used to handle the result of an MPI communication operation
     * asynchronously.
     */
    class MultiFuture : public Communicator::MultiFuture {
        friend class MPI;

      public:
        /**
         * @brief Construct a MultiFuture.
         *
         * @param req The MPI request handle for the operation.
         * @param data A shared pointer to the data buffer.
         * @warning It is undefined behaviour to create such a future
         * for a receive operation. It should only be done for send
         * operations when sending the same data to multiple
         * recipients.
         */
        MultiFuture(MPI_Request req, std::shared_ptr<Buffer> data)
            : req_{req}, data_{std::move(data)} {}

        ~MultiFuture() noexcept override = default;

      private:
        using Base = Communicator::MultiFuture;

        MPI_Request req_;  ///< The MPI request associated with the operation.
        std::shared_ptr<Buffer> data_;  ///< The data buffer.
    };

    /**
     * @brief Construct an MPI communicator.
     *
     * @param comm The MPI communicator to be used for communication.
     * @param options Configuration options.
     */
    MPI(MPI_Comm comm, config::Options options);

    ~MPI() noexcept override = default;

    /**
     * @copydoc Communicator::rank
     */
    [[nodiscard]] Rank rank() const override {
        return rank_;
    }

    /**
     * @copydoc Communicator::nranks
     */
    [[nodiscard]] Rank nranks() const override {
        return nranks_;
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
     */
    // clang-format on
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, Tag tag
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::send(std::unique_ptr<Buffer>, std::span<Rank> const, Tag tag)
     */
    // clang-format on
    [[nodiscard]] std::vector<std::unique_ptr<Communicator::MultiFuture>> send(
        std::unique_ptr<Buffer> msg, std::span<Rank> const destinations, Tag tag
    ) override;

    /**
     * @copydoc Communicator::recv
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> recv(
        Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
    ) override;

    /**
     * @copydoc Communicator::recv_any
     */
    [[nodiscard]] std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(
        Tag tag
    ) override;

  private:
    /**
     * @brief Test for completion of multiple future objects.
     * @tparam Type of future being tested.
     *
     * @param[inout] futures Futures to be tested.
     * @return Completed futures (erased from the input vector).
     */
    template <typename Future>
        requires std::is_same_v<Future, MPI::Future>
                 || std::is_same_v<Future, MPI::MultiFuture>
    std::vector<std::unique_ptr<typename Future::Base>> test_some(
        std::vector<std::unique_ptr<typename Future::Base>>& futures
    ) {
        if (futures.empty()) {
            return {};
        }
        std::vector<MPI_Request> reqs;
        reqs.reserve(futures.size());
        for (auto const& future : futures) {
            auto mpi_future = dynamic_cast<Future const*>(future.get());
            RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
            reqs.push_back(mpi_future->req_);
        }

        // Get completed requests as indices into `futures` (and `reqs`).
        std::vector<int> indices(reqs.size());
        int num_completed{0};
        RAPIDSMPF_MPI(MPI_Testsome(
            reqs.size(), reqs.data(), &num_completed, indices.data(), MPI_STATUSES_IGNORE
        ));
        RAPIDSMPF_EXPECTS(
            num_completed != MPI_UNDEFINED, "Expected at least one active handle."
        );
        if (num_completed == 0) {
            return {};
        }
        std::vector<std::unique_ptr<typename Future::Base>> completed;
        completed.reserve(static_cast<std::size_t>(num_completed));
        std::ranges::transform(
            indices.begin(),
            indices.begin() + num_completed,
            std::back_inserter(completed),
            [&](std::size_t i) { return std::move(futures[i]); }
        );
        std::erase(futures, nullptr);
        return completed;
    }

  public:
    /**
     * @copydoc Communicator::test_some
     */
    std::vector<std::unique_ptr<Communicator::Future>> test_some(
        std::vector<std::unique_ptr<Communicator::Future>>& future_vector
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::test_some(std::vector<std::unique_ptr<MultiFuture>>&)
     */
    // clang-format on
    std::vector<std::unique_ptr<Communicator::MultiFuture>> test_some(
        std::vector<std::unique_ptr<Communicator::MultiFuture>>& future_vector
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::test_some(std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const& future_map)
     */
    // clang-format on
    std::vector<std::size_t> test_some(
        std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
            future_map
    ) override;

    /**
     * @copydoc Communicator::wait
     */
    [[nodiscard]] std::unique_ptr<Buffer> wait(
        std::unique_ptr<Communicator::Future> future
    ) override;

    /**
     * @copydoc Communicator::wait_all
     */
    [[nodiscard]] std::vector<std::unique_ptr<Buffer>> wait_all(
        std::vector<std::unique_ptr<Communicator::Future>>&& futures
    ) override;

    /**
     * @copydoc Communicator::get_gpu_data
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
    MPI_Comm comm_;
    Rank rank_;
    Rank nranks_;
    Logger logger_;
};


}  // namespace rapidsmpf
