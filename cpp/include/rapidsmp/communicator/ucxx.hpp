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

struct UCXXSharedResources {
    std::shared_ptr<::ucxx::Listener> listener_{nullptr};
    Rank rank_{Rank(-1)};
    Rank next_rank_{1};
    // TODO: We probably need to make endpoints thread-safe.
    EndpointsMap endpoints_{};
    RankToEndpointMap rank_to_endpoint_{};
    RankToListenerAddressMap rank_to_listener_address_{};
    const ::ucxx::AmReceiverCallbackInfo control_callback_info_{
        ::ucxx::AmReceiverCallbackInfo("rapidsmp", 0)
    };
    std::vector<std::unique_ptr<HostFuture>> futures_{
        std::vector<std::unique_ptr<HostFuture>>()
    };

    UCXXSharedResources() = delete;

    UCXXSharedResources(bool root) : rank_(Rank(root ? 0 : -1)) {}

    void set_rank(Rank rank) {
        rank_ = rank;
    }

    Rank get_next_worker_rank() {
        if (rank_ != 0)
            throw std::runtime_error("This method can only be called by rank 0");
        return next_rank_++;
    }

    void register_listener(std::shared_ptr<::ucxx::Listener> listener) {
        listener_ = listener;
        auto listener_address = ListenerAddress{
            .host = listener->getIp(), .port = listener->getPort(), .rank = rank_
        };
        rank_to_listener_address_[rank_] = listener_address;
    }

    void register_endpoint(std::shared_ptr<::ucxx::Endpoint> endpoint) {
        endpoints_[endpoint->getHandle()] = endpoint;
    }

    void register_endpoint(const Rank rank, const ucp_ep_h endpoint_handle) {
        auto endpoint = endpoints_[endpoint_handle];
        rank_to_endpoint_[rank] = endpoint;
    }

    void register_endpoint(const Rank rank, std::shared_ptr<::ucxx::Endpoint> endpoint) {
        rank_to_endpoint_[rank] = endpoint;
        endpoints_[endpoint->getHandle()] = endpoint;
    }

    std::shared_ptr<::ucxx::Listener> get_listener() {
        return listener_;
    }

    std::shared_ptr<::ucxx::Endpoint> get_endpoint(const ucp_ep_h ep_handle) {
        return endpoints_.at(ep_handle);
    }

    std::shared_ptr<::ucxx::Endpoint> get_endpoint(const Rank rank) {
        return rank_to_endpoint_.at(rank);
    }

    ListenerAddress get_listener_address(const Rank rank) {
        return rank_to_listener_address_.at(rank);
    }

    void register_listener_address(
        const Rank rank, const ListenerAddress listener_address
    ) {
        rank_to_listener_address_[rank] = listener_address;
    }

    void add_future(std::unique_ptr<HostFuture> future) {
        futures_.push_back(std::move(future));
    }
};

enum class ControlMessage {
    AssignRank = 0,  //< Root assigns a rank to incoming client connection
    RegisterRank,  ///< Inform rank to remote process (non-root) after endpoint is
                   ///< established
    QueryListenerAddress,  ///< Ask for the remote endpoint's listener address
    ReplyListenerAddress  ///< Reply to `QueryListenerAddress` with the listener address
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
        return shared_resources_->rank_;
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

    void barrier();

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
    std::shared_ptr<UCXXSharedResources> shared_resources_;
    std::uint32_t nranks_;
    Logger logger_;

    std::shared_ptr<::ucxx::Endpoint> get_endpoint(Rank rank);
};


}  // namespace rapidsmp
