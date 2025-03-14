/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <ucxx/api.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

namespace ucxx {


using HostPortPair =
    std::pair<std::string, uint16_t>;  ///< A string with hostname or IP address, and the
                                       ///< port a listener is bound to.
using RemoteAddress = std::variant<
    HostPortPair,
    std::shared_ptr<::ucxx::Address>>;  ///< Host/port pair or worker address identifying
                                        ///< the remote UCXX listener or worker.

/**
 * @brief The progress mode to use with UCXX worker.
 */
enum class ProgressMode {
    Blocking = 0,
    Polling,
    ThreadBlocking,
    ThreadPolling
};

/**
 * @brief Storage for a listener address.
 *
 * Stores a listener address, composed of the hostname or IP address, port and rank the
 * listener corresponds to.
 */
class ListenerAddress {
  public:
    RemoteAddress address;  ///< Hostname/port pair or UCXX address.
    Rank rank{};  ///< The rank of the listener.
};

class SharedResources;

/**
 * @brief A UCXX initialized rank.
 *
 * This class is a container returned by the `init()` function with an opaque object
 * to an initialized UCXX rank. An object of this class is later used to initialize
 * a `UCXX` object.
 */
class InitializedRank {
  public:
    /**
     * @brief Construct an initialized UCXX rank.
     *
     * Construct an initialized UCXX rank.
     *
     * @param shared_resources Opaque object created by `init()`.
     */
    InitializedRank(std::shared_ptr<SharedResources> shared_resources);

    std::shared_ptr<SharedResources> shared_resources_{nullptr
    };  ///< Opaque object created by `init()`.
};

/**
 * @brief Initialize the current process with a UCXX rank.
 *
 * Initialize the current process with a UCXX rank, returning an opaque object that is
 * later used to initialize a `UCXX` object.
 *
 * @param worker The UCXX worker, or nullptr to create one internally.
 * @param nranks The number of ranks requested for the cluster.
 * @param remote_address Host/port pair or worker address identifying the remote UCXX
 *                       listener or worker. Used only by non-root ranks to connect to a
 *                       previously initialized root rank, for which the default
 *                       `std::nullopt` is specified.
 * @param progress_mode The progress mode to use with the UCXX worker.
 *
 * @throws std::logic_error if the `remote_address` is an invalid object.
 */
std::unique_ptr<rapidsmp::ucxx::InitializedRank> init(
    std::shared_ptr<::ucxx::Worker> worker,
    std::uint32_t nranks,
    std::optional<RemoteAddress> remote_address = std::nullopt,
    ProgressMode progress_mode = ProgressMode::ThreadBlocking
);

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
     * @brief Construct the UCXX rank.
     *
     * Construct the UCXX rank using the context previously returned from the call to
     * `init()`.
     *
     * @param ucxx_initialized_rank The previously initialized UCXX rank.
     */
    UCXX(std::unique_ptr<InitializedRank> ucxx_initialized_rank);

    ~UCXX() noexcept override;

    /**
     * @copydoc Communicator::rank
     */
    [[nodiscard]] Rank rank() const override;

    /**
     * @copydoc Communicator::nranks
     */
    [[nodiscard]] int nranks() const override;

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

    /**
     * @brief Barrier to synchronize all ranks
     *
     * The barrier is not intended to be performant and therefore should not be
     * used as part of regular rapidsmp logic, it is designed as a mechanism to
     * wait for the cluster to bootstrap and to wait for completion of all tasks.
     */
    void barrier();

    /**
     * @brief Get address to which listener is bound.
     *
     * @return The address to which listener is bound.
     */
    ListenerAddress listener_address();

  private:
    std::shared_ptr<SharedResources> shared_resources_;
    Logger logger_;

    std::shared_ptr<::ucxx::Endpoint> get_endpoint(Rank rank);
    void progress_worker();
};

}  // namespace ucxx

}  // namespace rapidsmp
