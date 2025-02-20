/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include <mpi.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

/**
 * @namespace rapidsmp::mpi
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
#define RAPIDSMP_MPI(call) \
    rapidsmp::mpi::detail::check_mpi_error((call), __FILE__, __LINE__)

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
        MPI_Request req_;  ///< The MPI request associated with the operation.
        std::unique_ptr<Buffer> data_;  ///< The data buffer.
    };

    /**
     * @brief Construct an MPI communicator.
     *
     * @param comm The MPI communicator to be used for communication.
     */
    MPI(MPI_Comm comm);

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
    [[nodiscard]] int nranks() const override {
        return nranks_;
    }

    /**
     * @copydoc Communicator::send
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<std::vector<uint8_t>> msg,
        Rank rank,
        Tag tag,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::send(std::unique_ptr<Buffer> msg, Rank rank, Tag tag, rmm::cuda_stream_view stream)
     */
    // clang-format on
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, Tag tag, rmm::cuda_stream_view stream
    ) override;

    /**
     * @copydoc Communicator::recv
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> recv(
        Rank rank,
        Tag tag,
        std::unique_ptr<Buffer> recv_buffer,
        rmm::cuda_stream_view stream
    ) override;

    /**
     * @copydoc Communicator::recv_any
     */
    [[nodiscard]] std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(Tag tag
    ) override;

    /**
     * @copydoc Communicator::test_some
     */
    std::vector<std::size_t> test_some(
        std::vector<std::unique_ptr<Communicator::Future>> const& future_vector
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
    std::uint32_t nranks_;
    Logger logger_;
};


}  // namespace rapidsmp
