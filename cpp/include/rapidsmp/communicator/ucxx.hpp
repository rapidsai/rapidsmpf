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
#include <variant>
#include <vector>

#include <ucxx/api.h>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmp/communicator/communicator.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

/**
 * @namespace rapidsmp::ucxx
 * @brief Collection of helpful [UCXX](https://github.com/rapidsai/ucxx/) functions.
 */
namespace ucxx {

/**
 * @brief Helper to initialize UCXX with threading support.
 *
 * @param argc Pointer to the number of arguments passed to the program.
 * @param argv Pointer to the argument vector passed to the program.
 */
// void init(int* argc, char*** argv);

/**
 * @brief Helper to check the UCXX errcode of a UCXX call.
 *
 * A macro to check the result of a UCXX call and handle any error codes that are
 * returned.
 *
 * @param call The UCXX call to be checked for errors.
 */
/*
#define RAPIDSMP_MPI(call) \
    rapidsmp::mpi::detail::check_mpi_error((call), __FILE__, __LINE__)
*/

namespace detail {
/**
 * @brief Checks and reports UCXX error codes.
 *
 * @param error_code The UCXX error code to check.
 * @param file The file where the UCXX call occurred.
 * @param line The line number where the UCXX call occurred.
 */
void check_ucxx_error(int error_code, const char* file, int line);
}  // namespace detail
}  // namespace ucxx

using EndpointsMap = std::unordered_map<ucp_ep_h, std::shared_ptr<::ucxx::Endpoint>>;

struct ListenerAddress {
    std::string host{};
    uint16_t port{};
    Rank rank{};
};

using RankToEndpointMap = std::unordered_map<Rank, std::shared_ptr<::ucxx::Endpoint>>;
using RankToListenerAddressMap = std::unordered_map<Rank, ListenerAddress>;

class HostFuture {
    friend class UCXX;

  public:
    /**
     * @brief Construct a Future.
     *
     * @param req The UCXX request handle for the operation.
     * @param data A unique pointer to the data buffer.
     */
    HostFuture(
        std::shared_ptr<::ucxx::Request> req, std::unique_ptr<std::vector<uint8_t>> data
    )
        : req_{std::move(req)}, data_{std::move(data)} {}

    ~HostFuture() noexcept = default;

    HostFuture(HostFuture&&) = default;  ///< Movable.
    HostFuture(HostFuture&) = delete;  ///< Not copyable.

  private:
    std::shared_ptr<::ucxx::Request>
        req_;  ///< The UCXX request associated with the operation.
    std::unique_ptr<std::vector<uint8_t>> data_;  ///< The data buffer.
};

struct ListenerContainer {
    std::shared_ptr<::ucxx::Listener> listener_{nullptr};
    std::shared_ptr<Rank> rank_{nullptr};
    // TODO: We probably need to make endpoints thread-safe.
    std::shared_ptr<EndpointsMap> endpoints_{nullptr};
    std::shared_ptr<RankToEndpointMap> rank_to_endpoint_{nullptr};
    std::shared_ptr<RankToListenerAddressMap> rank_to_listener_address_{nullptr};
    bool root_;
    std::shared_ptr<std::vector<std::unique_ptr<HostFuture>>> futures_{nullptr};
    std::function<Rank()> get_next_worker_rank_;
    // std::function<Communicator::Logger&()> logger;
    // Communicator::Logger& log;
};

enum class ControlMessage {
    AssignRank = 0,
    RegisterEndpoint,
    SetListenerAddress,
    GetListenerAddress
};
using ControlData = std::variant<Rank, ListenerAddress>;

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
     * @brief Construct a UCXX communicator.
     *
     * @param comm The UCXX communicator to be used for communication.
     */
    UCXX(std::shared_ptr<::ucxx::Worker> worker, std::uint32_t nranks);
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
    [[nodiscard]] Rank rank() const override {
        return *rank_;
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

    void barrier() override;

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
    std::shared_ptr<::ucxx::Worker> worker_;
    std::shared_ptr<::ucxx::Listener> listener_;
    std::shared_ptr<ListenerContainer> listener_container_;
    std::shared_ptr<EndpointsMap> endpoints_;
    std::shared_ptr<RankToEndpointMap> rank_to_endpoint_;
    std::shared_ptr<RankToListenerAddressMap> rank_to_listener_address_;
    std::shared_ptr<Rank> rank_;
    std::uint32_t nranks_;
    Rank next_rank_;
    Logger logger_;

    Rank get_next_worker_rank();
};


}  // namespace rapidsmp
