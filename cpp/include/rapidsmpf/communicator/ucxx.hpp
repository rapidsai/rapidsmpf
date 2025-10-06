/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdlib>
#include <memory>
#include <utility>

#include <ucxx/api.h>

#include <rmm/device_buffer.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

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
enum class ProgressMode : std::uint8_t {
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

    std::shared_ptr<SharedResources> shared_resources_{
        nullptr
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
 * @param options The options to use for the communicator, currently supports only
 *                `"ucxx_progress_mode"`.
 *
 * @throws std::logic_error if the `remote_address` is an invalid object.
 */
std::unique_ptr<rapidsmpf::ucxx::InitializedRank> init(
    std::shared_ptr<::ucxx::Worker> worker,
    Rank nranks,
    std::optional<RemoteAddress> remote_address,
    config::Options options
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
         * @brief Construct a Future from a data buffer.
         *
         * @param req The UCXX request handle for the operation.
         * @param data_buffer A unique pointer to the data buffer.
         */
        Future(std::shared_ptr<::ucxx::Request> req, std::unique_ptr<Buffer> data_buffer)
            : req_{std::move(req)}, data_buffer_{std::move(data_buffer)} {}

        /**
         * @brief Construct a Future from synchronized host data.
         *
         * This constructor is used for UCXX operations where the data resides
         * in host memory and is guaranteed to be valid at the time of the call.
         *
         * @param req The UCXX request handle for the operation.
         * @param synced_host_data A unique pointer to a vector containing host memory.
         */
        Future(
            std::shared_ptr<::ucxx::Request> req,
            std::unique_ptr<std::vector<uint8_t>> synced_host_data
        )
            : req_{std::move(req)}, synced_host_data_{std::move(synced_host_data)} {}

        ~Future() noexcept override = default;

      private:
        std::shared_ptr<::ucxx::Request> req_;
        std::unique_ptr<Buffer> data_buffer_;
        // Dedicated storage for host data that is valid at the time of construction.
        std::unique_ptr<std::vector<uint8_t>> synced_host_data_;
    };

    /**
     * @brief Construct the UCXX rank.
     *
     * Construct the UCXX rank using the context previously returned from the call to
     * `init()`.
     *
     * @param ucxx_initialized_rank The previously initialized UCXX rank.
     * @param options Configuration options.
     */
    UCXX(std::unique_ptr<InitializedRank> ucxx_initialized_rank, config::Options options);

    ~UCXX() noexcept override;

    /**
     * @copydoc Communicator::rank
     */
    [[nodiscard]] Rank rank() const override;

    /**
     * @copydoc Communicator::nranks
     */
    [[nodiscard]] Rank nranks() const override;

    /**
     * @copydoc Communicator::send
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<std::vector<uint8_t>> msg, Rank rank, Tag tag
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::send(std::unique_ptr<Buffer> msg, Rank rank, Tag tag)
     */
    // clang-format on
    [[nodiscard]] std::unique_ptr<Communicator::Future> send(
        std::unique_ptr<Buffer> msg, Rank rank, Tag tag
    ) override;

    /**
     * @copydoc Communicator::recv
     */
    [[nodiscard]] std::unique_ptr<Communicator::Future> recv(
        Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
    ) override;

    // clang-format off
    /**
     * @copydoc Communicator::recv_sync_host_data(Rank rank, Tag tag, std::unique_ptr<std::vector<uint8_t>> synced_buffer)
     */
    // clang-format on
    [[nodiscard]] std::unique_ptr<Communicator::Future> recv_sync_host_data(
        Rank rank, Tag tag, std::unique_ptr<std::vector<uint8_t>> synced_buffer
    ) override;

    /**
     * @copydoc Communicator::recv_any
     *
     * @throws ucxx::Error if there is a message but the receive does not complete
     * successfully.
     */
    [[nodiscard]] std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> recv_any(
        Tag tag
    ) override;

    /**
     * @copydoc Communicator::recv_from
     *
     * @throws ucxx::Error if there is a message but the receive does not complete
     * successfully.
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> recv_from(
        Rank src, Tag tag
    ) override;

    /**
     * @copydoc Communicator::test_some
     *
     * @throws ucxx::Error if any completed futures did not complete successfully.
     */
    std::pair<
        std::vector<std::unique_ptr<Communicator::Future>>,
        std::vector<std::size_t>>
    test_some(std::vector<std::unique_ptr<Communicator::Future>>& future_vector) override;

    // clang-format off
    /**
     * @copydoc Communicator::test_some(std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const& future_map)
     *
     * @throws ucxx::Error if any completed futures did not complete successfully.
     */
    // clang-format on
    std::vector<std::size_t> test_some(
        std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
            future_map
    ) override;

    /**
     * @copydoc Communicator::wait
     *
     * @throws ucxx::Error if the future did not complete successfully.
     */
    [[nodiscard]] std::unique_ptr<Buffer> wait(
        std::unique_ptr<Communicator::Future> future
    ) override;

    /**
     * @copydoc Communicator::release_data
     */
    [[nodiscard]] std::unique_ptr<Buffer> release_data(
        std::unique_ptr<Communicator::Future> future
    ) override;

    /**
     * @copydoc Communicator::release_sync_host_data
     */
    [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> release_sync_host_data(
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
     * used as part of regular rapidsmpf logic, it is designed as a mechanism to
     * wait for the cluster to bootstrap and to wait for completion of all tasks.
     */
    void barrier();

    /**
     * @brief Get address to which listener is bound.
     *
     * @return The address to which listener is bound.
     */
    ListenerAddress listener_address();

    /**
     * @brief Creates a new communicator with a single rank.
     *
     * This method creates a new communicator that acts as if it was a single rank,
     * similar to MPI_Comm_split when color is the rank of the current process and key is
     * 0.
     *
     * @note This method is generally used for testing.
     *
     * @return A new UCXX communicator with a single rank.
     */
    std::shared_ptr<UCXX> split();

  private:
    std::shared_ptr<SharedResources> shared_resources_;
    config::Options options_;
    Logger logger_;

    std::shared_ptr<::ucxx::Endpoint> get_endpoint(Rank rank);
    void progress_worker();
};

}  // namespace ucxx

}  // namespace rapidsmpf
