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
#include <utility>
#include <vector>

#include <ucxx/api.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

class UCXXSharedResources;

/**
 * @brief UCXX communicator class that implements the `Communicator` interface.
 *
 * This class implements communication functions using UCXX, allowing for data exchange
 * between processes in a distributed system. It supports sending and receiving data, both
 * on the CPU and GPU, and provides asynchronous operations with support for future
 * results.
 */
class UCXX final : public Communicator {
  public:
    /**
     * @brief Represents the future result of an UCXX operation.
     *
     * This class is used to handle the result of an UCXX communication operation
     * asynchronously.
     */
    class Future : public Communicator::Future {
        friend class UCXX;

      public:
        /**
         * @brief Construct a Future.
         *
         * @param req The UCXX request handle for the operation.
         * @param data A unique pointer to the data buffer.
         */
        Future(std::shared_ptr<::ucxx::Request> req, std::unique_ptr<Buffer> data)
            : req_{std::move(req)}, data_{std::move(data)} {}

        ~Future() noexcept override = default;

      private:
        std::shared_ptr<::ucxx::Request>
            req_;  ///< The UCXX request associated with the operation.
        std::unique_ptr<Buffer> data_;  ///< The data buffer.
    };

    /**
     * @brief Construct the root UCXX communicator.
     *
     * @param worker The UCXX worker, or nullptr to create one internally.
     * @param nranks The number of ranks requested for the cluster.
     */
    UCXX(std::shared_ptr<::ucxx::Worker> worker, std::uint32_t nranks);

    /**
     * @brief Construct additional (non-root) UCXX communicator.
     *
     * @param worker The UCXX worker, or nullptr to create one internally.
     * @param nranks The number of ranks requested for the cluster.
     * @param root_host The hostname or IP address where the root rank is listening.
     * @param root_port The port where the root rank is listening.
     */
    UCXX(
        std::shared_ptr<::ucxx::Worker> worker,
        std::uint32_t nranks,
        std::string root_host,
        uint16_t root_port
    );

    ~UCXX() noexcept override;

    /**
     * @copydoc Communicator::rank
     */
    [[nodiscard]] Rank rank() const override;

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
        int tag,
        rmm::cuda_stream_view stream,
        BufferResource* br
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::send(std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream)
     */
    // clang-format on
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream
    ) override;

    /**
     * @copydoc Communicator::recv
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> recv(
        Rank rank,
        int tag,
        std::unique_ptr<Buffer> recv_buffer,
        rmm::cuda_stream_view stream
    ) override;

    /**
     * @copydoc Communicator::recv_any
     */
    [[nodiscard]] std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(int tag
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

    /**
     * @brief Barrier to synchronize all ranks
     *
     * The barrier is not intended to be performant and therefore should not be
     * used as part of regular rapidsmp logic, it is designed as a mechanism to
     * wait for the cluster to bootstrap and to wait for completion of all tasks.
     */
    void barrier();

  private:
    std::shared_ptr<::ucxx::Worker> worker_;
    std::shared_ptr<UCXXSharedResources> shared_resources_;
    std::uint32_t nranks_;
    Logger logger_;

    std::shared_ptr<::ucxx::Endpoint> get_endpoint(Rank rank);
    void progress_worker();
};


}  // namespace rapidsmp
