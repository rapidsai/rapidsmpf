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

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include <rapidsmp/communicator/ucxx.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

namespace ucxx {

namespace {

/**
 * @brief Control message types.
 *
 * There are 4 types of control messages:
 * 1. AssignRank: Sent by the root during `listener_callback` to each new
 * client (remote process) that connects to it. This message contains the rank
 * this client should respond to from now on.
 * 2. QueryListenerAddress: Asks the root rank for the listener address of
 * another rank, with the purpose of establishing an endpoint to that rank for
 * direct message transfers.
 *
 */
enum class ControlMessage {
    AssignRank = 0,  ///< Root assigns a rank to incoming client connection
    QueryRank,  ///< Ask root for a rank
    QueryListenerAddress,  ///< Ask for the remote endpoint's listener address
    RegisterRank,  ///< Inform rank to remote process (non-root) after endpoint is
                   ///< established
    ReplyListenerAddress  ///< Reply to `QueryListenerAddress` with the listener address
};

enum class ListenerAddressType {
    WorkerAddress = 0,
    HostPort,
    Undefined
};

using Rank = rapidsmp::Rank;
using HostPortPair = rapidsmp::ucxx::HostPortPair;
using RemoteAddress = rapidsmp::ucxx::RemoteAddress;
using ListenerAddress = rapidsmp::ucxx::ListenerAddress;
using ControlData = std::variant<Rank, ListenerAddress>;
using EndpointsMap = std::unordered_map<ucp_ep_h, std::shared_ptr<::ucxx::Endpoint>>;
using RankToEndpointMap = std::unordered_map<Rank, std::shared_ptr<::ucxx::Endpoint>>;
using RankToListenerAddressMap = std::unordered_map<Rank, ListenerAddress>;

class HostFuture {
    friend class UCXX;

  public:
    /**
     * @brief Construct a HostFuture.
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

    [[nodiscard]] bool completed() const {
        return req_->isCompleted();
    }

  private:
    std::shared_ptr<::ucxx::Request>
        req_;  ///< The UCXX request associated with the operation.
    std::unique_ptr<std::vector<uint8_t>> data_;  ///< The data buffer.
};

}  // namespace

class SharedResources {
  private:
    std::shared_ptr<::ucxx::Worker> worker_{nullptr};  ///< UCXX Listener
    std::shared_ptr<::ucxx::Listener> listener_{nullptr};  ///< UCXX Listener
    Rank rank_{Rank(-1)};  ///< Rank of the current process
    std::uint32_t nranks_{0};  ///< Rank of the current process
    std::atomic<Rank> next_rank_{1
    };  ///< Rank to assign for the next client that connects (root only)
    EndpointsMap endpoints_{};  ///< Map of UCP handle to UCXX endpoints of known ranks
    RankToEndpointMap rank_to_endpoint_{};  ///< Map of ranks to UCXX endpoints
    RankToListenerAddressMap rank_to_listener_address_{
    };  ///< Map of rank to listener addresses
    const ::ucxx::AmReceiverCallbackInfo control_callback_info_{
        ::ucxx::AmReceiverCallbackInfo("rapidsmp", 0)
    };  ///< UCXX callback info for control messages
    std::vector<std::unique_ptr<HostFuture>> futures_{
    };  ///< Futures to incomplete requests.
    std::vector<std::function<void()>> delayed_progress_callbacks_{};
    std::mutex endpoints_mutex_{};
    std::mutex futures_mutex_{};
    std::mutex listener_mutex_{};
    std::mutex delayed_progress_callbacks_mutex_{};

  public:
    UCXX::Logger* logger{nullptr};  ///< UCXX Listener

    /**
     * @brief Construct UCXX shared resources.
     *
     * Constructor UCXX shared resources, assigning the proper rank 0 for root,
     * other ranks must call `set_rank()` at the appropriate time.
     *
     * @param root Whether the rank is the root rank.
     * @param nranks The number of ranks requested for the cluster.
     */
    SharedResources(std::shared_ptr<::ucxx::Worker> worker, bool root, uint32_t nranks)
        : worker_{worker}, rank_{Rank(root ? 0 : -1)}, nranks_{nranks} {}

    SharedResources(SharedResources&&) = delete;  ///< Not movable.
    SharedResources(SharedResources&) = delete;  ///< Not copyable.

    /**
     * @brief Sets the rank of a non-root rank.
     *
     * Sets the rank of a non-root rank to the specified value.
     *
     * @param rank The rank to set.
     */
    void set_rank(Rank rank) {
        rank_ = rank;
    }

    /**
     * @brief Gets the rank.
     *
     * Returns the rank.
     *
     * @return The rank.
     */
    [[nodiscard]] Rank rank() const {
        return rank_;
    }

    /**
     * @brief Gets the number of ranks in the cluster.
     *
     * Returns the number of ranks in the cluster.
     *
     * @return The number of ranks in the cluster.
     */
    [[nodiscard]] std::uint32_t nranks() const {
        return nranks_;
    }

    /**
     * @brief Gets the next worker rank.
     *
     * Returns the next available worker rank. This method can only be called
     * by root rank (rank 0).
     *
     * @return The next available worker rank.
     * @throws std::logic_error If called by rank other than 0.
     */
    [[nodiscard]] Rank get_next_worker_rank() {
        RAPIDSMP_EXPECTS(rank_ == 0, "This method can only be called by rank 0");
        return next_rank_++;
    }

    /**
     * @brief Gets the UCXX worker.
     *
     * Returns the UCXX worker.
     *
     * @return The UCXX worker.
     */
    [[nodiscard]] std::shared_ptr<::ucxx::Worker> get_worker() {
        return worker_;
    }

    [[nodiscard]] ::ucxx::AmReceiverCallbackInfo get_control_callback_info() const {
        return control_callback_info_;
    }

    /**
     * @brief Registers a listener.
     *
     * Registers a listener with the UCXX shared resources.
     *
     * @param listener The listener to register.
     */
    void register_listener(std::shared_ptr<::ucxx::Listener> listener) {
        std::lock_guard<std::mutex> lock(listener_mutex_);
        auto worker = std::dynamic_pointer_cast<::ucxx::Worker>(listener->getParent());
        rank_to_listener_address_[rank_] =
            ListenerAddress{worker->getAddress(), .rank = rank_};
        listener_ = std::move(listener);
    }

    /**
     * @brief Registers an endpoint for a specific rank.
     *
     * Registers an endpoint and associate it with a specific rank.
     *
     * @param rank The rank to register the endpoint for.
     * @param endpoint The endpoint to register.
     */
    void register_endpoint(Rank const rank, std::shared_ptr<::ucxx::Endpoint> endpoint) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        rank_to_endpoint_[rank] = endpoint;
        endpoints_[endpoint->getHandle()] = std::move(endpoint);
    }

    /**
     * @brief Registers an endpoint without a rank.
     *
     * Registers an endpoint to a remote process with a still unknown rank.
     * The rank must be later associated by calling the `associate_endpoint_rank()`.
     *
     * @param endpoint The endpoint to register.
     */
    void register_endpoint(std::shared_ptr<::ucxx::Endpoint> endpoint) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        endpoints_[endpoint->getHandle()] = std::move(endpoint);
    }

    /**
     * @brief Associate endpoint with a specific rank.
     *
     * Associate a previously registered endpoint to a specific rank.
     *
     * @param rank The rank to register the endpoint for.
     * @param endpoint_handle The handle of the endpoint to register.
     */
    void associate_endpoint_rank(Rank const rank, ucp_ep_h const endpoint_handle) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        rank_to_endpoint_[rank] = endpoints_[endpoint_handle];
    }

    /**
     * @brief Gets the listener.
     *
     * Returns the registered listener.
     *
     * @return The registered listener.
     */
    [[nodiscard]] std::shared_ptr<::ucxx::Listener> get_listener() {
        std::lock_guard<std::mutex> lock(listener_mutex_);
        return listener_;
    }

    /**
     * @brief Gets an endpoint by handle.
     *
     * Returns the endpoint associated with the specified handle.
     *
     * @param ep_handle The handle of the endpoint to retrieve.
     * @return The endpoint associated with the specified handle.
     */
    [[nodiscard]] std::shared_ptr<::ucxx::Endpoint> get_endpoint(ucp_ep_h const ep_handle
    ) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        return endpoints_.at(ep_handle);
    }

    /**
     * @brief Gets an endpoint for a specific rank.
     *
     * Returns the endpoint associated with the specified rank.
     *
     * @param rank The rank to retrieve the endpoint for.
     * @return The endpoint associated with the specified rank.
     */
    [[nodiscard]] std::shared_ptr<::ucxx::Endpoint> get_endpoint(Rank const rank) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        return rank_to_endpoint_.at(rank);
    }

    /**
     * @brief Gets the listener address for a specific rank.
     *
     * Returns the listener address associated with the specified rank.
     *
     * @param rank The rank to retrieve the listener address for.
     * @return The listener address associated with the specified rank.
     */
    [[nodiscard]] ListenerAddress get_listener_address(Rank const rank) {
        std::lock_guard<std::mutex> lock(listener_mutex_);
        return rank_to_listener_address_.at(rank);
    }

    /**
     * @brief Registers a listener address for a specific rank.
     *
     * Registers a listener address and associated it with a specific rank.
     *
     * @param rank The rank to register the listener address for.
     * @param listener_address The listener address to register.
     */
    void register_listener_address(Rank const rank, ListenerAddress listener_address) {
        std::lock_guard<std::mutex> lock(listener_mutex_);
        rank_to_listener_address_[rank] = std::move(listener_address);
    }

    /**
     * @brief Adds a future to the list of incomplete requests.
     *
     * Adds a future to the list of incomplete requests.
     *
     * @param future The future to add.
     */
    void add_future(std::unique_ptr<HostFuture> future) {
        std::lock_guard<std::mutex> lock(futures_mutex_);
        futures_.push_back(std::move(future));
    }

    void barrier() {
        // The root needs to have endpoints to all other ranks to continue.
        while (rank_ == 0
               && rank_to_endpoint_.size() != static_cast<uint64_t>(nranks() - 1))
        {
            progress_worker();
        }

        if (rank_ == 0) {
            std::vector<std::shared_ptr<::ucxx::Request>> requests;
            for (auto& rank_to_endpoint : rank_to_endpoint_) {
                auto& endpoint = rank_to_endpoint.second;
                requests.push_back(endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST));
            }
            while (std::any_of(requests.cbegin(), requests.cend(), [](auto const& req) {
                return !req->isCompleted();
            }))
                progress_worker();

            requests.clear();

            for (auto& rank_to_endpoint : rank_to_endpoint_) {
                auto& endpoint = rank_to_endpoint.second;
                requests.push_back(endpoint->amRecv());
            }
            while (std::any_of(requests.cbegin(), requests.cend(), [](auto const& req) {
                return !req->isCompleted();
            }))
                progress_worker();
        } else {
            auto endpoint = get_endpoint(0);

            auto req = endpoint->amRecv();
            while (!req->isCompleted())
                progress_worker();

            req = endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST);
            while (!req->isCompleted())
                progress_worker();
        }
    }

    void clear_completed_futures() {
        std::lock_guard<std::mutex> lock(futures_mutex_);
        futures_.erase(
            std::remove_if(
                futures_.begin(),
                futures_.end(),
                [](std::unique_ptr<HostFuture> const& element) {
                    return element->completed();
                }
            ),
            futures_.end()
        );
    }

    void progress_worker() {
        decltype(delayed_progress_callbacks_) delayed_progress_callbacks{};
        {
            std::lock_guard<std::mutex> lock(delayed_progress_callbacks_mutex_);
            std::swap(delayed_progress_callbacks, delayed_progress_callbacks_);
        }
        for (auto& callback : delayed_progress_callbacks)
            callback();
        if (!worker_->isProgressThreadRunning()) {
            worker_->progress();
            // TODO: Support blocking progress mode
        }

        clear_completed_futures();
    }

    /**
     * @brief Adds a future to the list of incomplete requests.
     *
     * Adds a future to the list of incomplete requests.
     *
     * @param future The future to add.
     */
    void add_delayed_progress_callback(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(delayed_progress_callbacks_mutex_);
        delayed_progress_callbacks_.push_back(callback);
    }
};

namespace {

size_t get_size(ControlData const& data) {
    return std::visit(
        [](auto const& data) { return sizeof(std::decay_t<decltype(data)>); }, data
    );
}

void encode(void* dest, void const* src, size_t bytes, size_t& offset) {
    memcpy(static_cast<char*>(dest) + offset, src, bytes);
    offset += bytes;
}

void decode(void* dest, void const* src, size_t bytes, size_t& offset) {
    memcpy(dest, static_cast<char const*>(src) + offset, bytes);
    offset += bytes;
}

/**
 * @brief Pack listener address.
 *
 * Pack (i.e., serialize) `ListenerAddress` so that it may be sent over the wire.
 *
 * @param listener_address the listener address to pack.
 *
 * @return vector of bytes to be sent over the wire.
 */
std::unique_ptr<std::vector<uint8_t>> listener_address_pack(
    ListenerAddress const& listener_address
) {
    return std::visit(
        [&listener_address](auto&& remote_address) {
            size_t offset{0};
            std::unique_ptr<std::vector<uint8_t>> packed{nullptr};
            using T = std::decay_t<decltype(remote_address)>;
            if constexpr (std::is_same_v<T, HostPortPair>) {
                auto type = ListenerAddressType::HostPort;
                auto host_size = remote_address.first.size();
                size_t const total_size = sizeof(type) + sizeof(host_size) + host_size
                                          + sizeof(remote_address.second)
                                          + sizeof(listener_address.rank);
                auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

                auto encode_ = [&offset, &packed](void const* data, size_t bytes) {
                    encode(packed->data(), data, bytes, offset);
                };

                encode_(&type, sizeof(type));
                encode_(&host_size, sizeof(host_size));
                encode_(remote_address.first.data(), host_size);
                encode_(&remote_address.second, sizeof(remote_address.second));
                encode_(&listener_address.rank, sizeof(listener_address.rank));
                return packed;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<::ucxx::Address>>) {
                auto type = ListenerAddressType::WorkerAddress;
                auto address_size = remote_address->getLength();
                size_t const total_size = sizeof(type) + sizeof(address_size)
                                          + address_size + sizeof(listener_address.rank);
                auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

                auto encode_ = [&offset, &packed](void const* data, size_t bytes) {
                    encode(packed->data(), data, bytes, offset);
                };

                encode_(&type, sizeof(type));
                encode_(&address_size, sizeof(address_size));
                encode_(remote_address->getString().data(), address_size);
                encode_(&listener_address.rank, sizeof(listener_address.rank));
                return packed;
            } else {
                RAPIDSMP_EXPECTS(false, "Unknown argument type");
            }
        },
        listener_address.address
    );
}

/**
 * @brief Unpack control message.
 *
 * Unpack (i.e., deserialize) `ListenerAddress` received over the wire.
 *
 * @param packed vector of bytes that was received over the wire.
 *
 * @return listener address contained in the packed message.
 */
ListenerAddress listener_address_unpack(std::unique_ptr<std::vector<uint8_t>> packed) {
    size_t offset{0};

    auto decode_ = [&offset, &packed](void* data, size_t bytes) {
        decode(data, packed->data(), bytes, offset);
    };

    auto type = ListenerAddressType::Undefined;
    decode_(&type, sizeof(type));

    if (type == ListenerAddressType::WorkerAddress) {
        size_t address_size;
        decode_(&address_size, sizeof(address_size));

        auto address = std::string(address_size, '\0');
        Rank rank{-1};

        decode_(address.data(), address_size);
        decode_(&rank, sizeof(rank));

        auto ret = ListenerAddress{::ucxx::createAddressFromString(address), rank};
        return ret;
    } else if (type == ListenerAddressType::HostPort) {
        size_t host_size;
        decode_(&host_size, sizeof(host_size));

        auto host = std::string(host_size, '\0');
        auto port = std::uint16_t{0};
        auto rank = Rank{-1};
        decode_(host.data(), host_size);

        decode_(&port, sizeof(port));
        decode_(&rank, sizeof(rank));

        return ListenerAddress{std::make_pair(host, port), rank};
    } else {
        RAPIDSMP_EXPECTS(false, "Wrong type");
    }
}

/**
 * @brief Pack control message.
 *
 * Pack (i.e., serialize) `ControlMessage` and `ControlData` so that it may be
 * sent over the wire.
 *
 * @param control type of control message.
 * @param data data associated with the control message.
 */
std::unique_ptr<std::vector<uint8_t>> control_pack(
    ControlMessage control, ControlData data
) {
    size_t offset{0};

    if (control == ControlMessage::AssignRank || control == ControlMessage::RegisterRank
        || control == ControlMessage::QueryListenerAddress)
    {
        size_t const total_size = sizeof(control) + get_size(data);
        auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

        auto encode_ = [&offset, &packed](void const* data, size_t bytes) {
            encode(packed->data(), data, bytes, offset);
        };

        encode_(&control, sizeof(control));

        auto rank = std::get<Rank>(data);
        encode_(&rank, sizeof(rank));
        return packed;
    } else if (control == ControlMessage::QueryRank || control == ControlMessage::ReplyListenerAddress)
    {
        auto listener_address = std::get<ListenerAddress>(data);
        auto packed_listener_address = listener_address_pack(listener_address);
        size_t packed_listener_address_size = packed_listener_address->size();

        size_t const total_size = sizeof(control) + sizeof(packed_listener_address_size)
                                  + packed_listener_address_size;
        auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

        auto encode_ = [&offset, &packed](void const* data, size_t bytes) {
            encode(packed->data(), data, bytes, offset);
        };

        encode_(&control, sizeof(control));
        encode_(&packed_listener_address_size, sizeof(packed_listener_address_size));
        encode_(packed_listener_address->data(), packed_listener_address_size);

        return packed;
    } else {
        RAPIDSMP_EXPECTS(false, "Invalid control type");
    }
};

/**
 * @brief Unpack control message.
 *
 * Unpack (i.e., deserialize) `ControlMessage` and `ControlData` received over
 * the wire, and appropriately handle contained data.
 *
 * 1. AssignRank: Calls `SharedResources->set_rank()` to set own rank.
 * 2. QueryListenerAddress: Get the previously-registered listener address for
 * the rank being requested and reply the requester with the listener address.
 * A future with the reply is stored via `SharedResources->add_future()`.
 * This is only executed by the root.
 * 3. RegisterRank: Associate the endpoint created during `listener_callback()`
 * with the rank informed by the client.
 * 4. ReplyListenerAddress: Handle reply from root rank, associating the
 * received listener address with the requested rank.
 *
 * @param buffer bytes received via UCXX containing the packed message.
 * @param ep the UCX handle from which the message was received.
 * @param shared_resources UCXX shared resources of the current rank to
 * properly register received data.
 */
void control_unpack(
    std::shared_ptr<::ucxx::Buffer> buffer,
    ucp_ep_h ep,
    std::shared_ptr<rapidsmp::ucxx::SharedResources> shared_resources
) {
    size_t offset{0};

    auto decode_ = [&offset, &buffer](void* data, size_t bytes) {
        decode(data, buffer->data(), bytes, offset);
    };

    ControlMessage control;
    decode_(&control, sizeof(ControlMessage));

    if (control == ControlMessage::AssignRank) {
        Rank rank{-1};
        decode_(&rank, sizeof(rank));
        shared_resources->set_rank(rank);
    } else if (control == ControlMessage::QueryRank) {
        Rank client_rank = shared_resources->get_next_worker_rank();

        size_t packed_listener_address_size;
        decode_(&packed_listener_address_size, sizeof(packed_listener_address_size));

        auto packed_listener_address =
            std::make_unique<std::vector<uint8_t>>(packed_listener_address_size);
        decode_(packed_listener_address->data(), packed_listener_address_size);

        ListenerAddress listener_address =
            listener_address_unpack(std::move(packed_listener_address));
        listener_address.rank = client_rank;
        shared_resources->register_listener_address(
            client_rank, std::move(listener_address)
        );

        // This block cannot be called directly from here since it makes requests
        // (creating endpoint and sending AM) that require progressing the UCX worker,
        // which isn't allowed from within the callback this is already running in.
        // Therefore we make it a callback that is registered with SharedResources
        // and executed before progressing the worker in the next loop.
        auto callback = [shared_resources, client_rank]() {
            auto worker_address = std::get<std::shared_ptr<::ucxx::Address>>(
                shared_resources->get_listener_address(client_rank).address
            );
            auto endpoint =
                shared_resources->get_worker()->createEndpointFromWorkerAddress(
                    worker_address, true
                );
            shared_resources->register_endpoint(client_rank, endpoint);

            auto packed_client_rank =
                control_pack(ControlMessage::AssignRank, client_rank);
            auto req = endpoint->amSend(
                packed_client_rank->data(),
                packed_client_rank->size(),
                UCS_MEMORY_TYPE_HOST,
                shared_resources->get_control_callback_info()
            );
            shared_resources->add_future(
                std::make_unique<HostFuture>(req, std::move(packed_client_rank))
            );
        };

        shared_resources->add_delayed_progress_callback(std::move(callback));
    } else if (control == ControlMessage::RegisterRank) {
        Rank rank;
        decode_(&rank, sizeof(rank));
        shared_resources->associate_endpoint_rank(rank, ep);
    } else if (control == ControlMessage::ReplyListenerAddress) {
        size_t packed_listener_address_size;
        decode_(&packed_listener_address_size, sizeof(packed_listener_address_size));
        auto packed_listener_address =
            std::make_unique<std::vector<uint8_t>>(packed_listener_address_size);
        decode_(packed_listener_address->data(), packed_listener_address_size);

        ListenerAddress listener_address =
            listener_address_unpack(std::move(packed_listener_address));

        shared_resources->register_listener_address(
            listener_address.rank, std::move(listener_address)
        );
    } else if (control == ControlMessage::QueryListenerAddress) {
        Rank rank;
        decode_(&rank, sizeof(rank));
        auto listener_address = shared_resources->get_listener_address(rank);
        auto endpoint = shared_resources->get_endpoint(ep);
        auto packed_listener_address =
            control_pack(ControlMessage::ReplyListenerAddress, listener_address);
        auto req = endpoint->amSend(
            packed_listener_address->data(),
            packed_listener_address->size(),
            UCS_MEMORY_TYPE_HOST,
            shared_resources->get_control_callback_info()
        );
        shared_resources->add_future(
            std::make_unique<HostFuture>(req, std::move(packed_listener_address))
        );
    }
};

/**
 * @brief Listener callback executed each time a new client connects.
 *
 * Listener callback used by all UCXX listeners that executes each time a new
 * client connects.
 *
 * For all ranks when a new client connects, an endpoint to it is established
 * and retained for future communication. The root rank will always send an
 * `ControlMessage::AssignRank` to each incoming client to let the client know
 * which rank it should now respond to in the communicator world. Other ranks
 * will wait for a message `ControlMessage::RegisterRank` from the client
 * immediately after the connection is established with the client's rank in
 * the world communicator, so that the current rank knows which rank is
 * associated with the new endpoint.
 */
void listener_callback(ucp_conn_request_h conn_request, void* arg) {
    auto shared_resources = reinterpret_cast<rapidsmp::ucxx::SharedResources*>(arg);

    ucp_conn_request_attr_t attr{};
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    auto status = ucp_conn_request_query(conn_request, &attr);
    if (status != UCS_OK) {
        if (shared_resources->logger)
            // TODO: this should be error, but error level doesn't exist.
            shared_resources->logger->warn("Failed to create endpoint to client");
        return;
    }

    std::array<char, INET6_ADDRSTRLEN> ip_str;
    std::array<char, INET6_ADDRSTRLEN> port_str;
    ::ucxx::utils::sockaddr_get_ip_port_str(
        &attr.client_address, ip_str.data(), port_str.data(), INET6_ADDRSTRLEN
    );
    if (shared_resources->logger)
        shared_resources->logger->info(
            "Server received a connection request from client at address ",
            ip_str.data(),
            ":",
            port_str.data()
        );

    auto endpoint = shared_resources->get_listener()->createEndpointFromConnRequest(
        conn_request, true
    );

    if (shared_resources->rank() == 0) {
        Rank client_rank = shared_resources->get_next_worker_rank();
        shared_resources->register_endpoint(client_rank, endpoint);
        auto packed_client_rank = control_pack(ControlMessage::AssignRank, client_rank);
        auto req = endpoint->amSend(
            packed_client_rank->data(),
            packed_client_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            shared_resources->get_control_callback_info()
        );
        shared_resources->add_future(
            std::make_unique<HostFuture>(req, std::move(packed_client_rank))
        );
    } else {
        // We don't know the rank of the client that connected, we'll register
        // by its handle and the client will send a control message informing
        // its rank
        shared_resources->register_endpoint(endpoint);
    }
}

/**
 * @brief Callback executed by UCXX progress thread to create/acquire CUDA context.
 */
void create_cuda_context_callback(void* callbackArg) {
    cudaFree(nullptr);
}

}  // namespace

InitializedRank::InitializedRank(
    std::shared_ptr<rapidsmp::ucxx::SharedResources> shared_resources
)
    : shared_resources_(shared_resources) {}

std::unique_ptr<rapidsmp::ucxx::InitializedRank> init(
    std::shared_ptr<::ucxx::Worker> worker,
    std::uint32_t nranks,
    std::optional<RemoteAddress> remote_address
) {
    auto create_worker = []() {
        auto context = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
        auto worker = context->createWorker(false);
        // TODO: Allow other modes
        worker->setProgressThreadStartCallback(create_cuda_context_callback, nullptr);
        worker->startProgressThread(true);
        return worker;
    };

    if (remote_address) {
        if (worker == nullptr) {
            worker = create_worker();
        }
        auto shared_resources =
            std::make_shared<rapidsmp::ucxx::SharedResources>(worker, false, nranks);

        // Create listener
        shared_resources->register_listener(
            worker->createListener(0, listener_callback, shared_resources.get())
        );
        auto listener = shared_resources->get_listener();

        auto control_callback = ::ucxx::AmReceiverCallbackType(
            [shared_resources](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
                auto buffer = req->getRecvBuffer();
                control_unpack(req->getRecvBuffer(), ep, shared_resources);
            }
        );

        worker->registerAmReceiverCallback(
            shared_resources->get_control_callback_info(), control_callback
        );

        // Connect to root
        // TODO: Enable when Logger can be created before the UCXX communicator object.
        // See https://github.com/rapidsai/rapids-multi-gpu/issues/65 .
        //
        // log.debug(
        //     "Connecting to root node at ",
        //     *root_host,
        //     ":",
        //     *root_port,
        //     ". Current rank: ",
        //     shared_resources->rank()
        // );
        auto endpoint = std::visit(
            [shared_resources](auto&& remote_address) {
                using T = std::decay_t<decltype(remote_address)>;
                if constexpr (std::is_same_v<T, HostPortPair>) {
                    return shared_resources->get_worker()->createEndpointFromHostname(
                        remote_address.first, remote_address.second, true
                    );
                } else if constexpr (std::is_same_v<T, std::shared_ptr<::ucxx::Address>>)
                {
                    auto root_endpoint =
                        shared_resources->get_worker()->createEndpointFromWorkerAddress(
                            remote_address, true
                        );

                    auto packed_listener_address_rank = control_pack(
                        ControlMessage::QueryRank,
                        ListenerAddress{
                            shared_resources->get_worker()->getAddress(),
                            .rank = shared_resources->rank()
                        }
                    );

                    auto listener_address_req = root_endpoint->amSend(
                        packed_listener_address_rank->data(),
                        packed_listener_address_rank->size(),
                        UCS_MEMORY_TYPE_HOST,
                        shared_resources->get_control_callback_info()
                    );
                    while (!listener_address_req->isCompleted())
                        shared_resources->progress_worker();

                    return root_endpoint;
                } else {
                    RAPIDSMP_EXPECTS(false, "Unknown argument type");
                }
            },
            *remote_address
        );
        shared_resources->register_endpoint(Rank(0), endpoint);

        // Get my rank
        while (shared_resources->rank() == Rank(-1)) {
            shared_resources->progress_worker();
        }
        // TODO: Enable when Logger can be created before the UCXX communicator object.
        // See https://github.com/rapidsai/rapids-multi-gpu/issues/65 .
        // log.debug("Assigned rank: ", shared_resources->rank());

        if (const HostPortPair* host_port_pair =
                std::get_if<HostPortPair>(&*remote_address))
        {
            RAPIDSMP_EXPECTS(host_port_pair != nullptr, "Invalid pointer");

            // Inform listener address
            ListenerAddress listener_address = ListenerAddress{
                std::make_pair(listener->getIp(), listener->getPort()),
                .rank = shared_resources->rank()
            };
            auto packed_listener_address =
                control_pack(ControlMessage::ReplyListenerAddress, listener_address);
            auto req = endpoint->amSend(
                packed_listener_address->data(),
                packed_listener_address->size(),
                UCS_MEMORY_TYPE_HOST,
                shared_resources->get_control_callback_info()
            );
            while (!req->isCompleted())
                shared_resources->progress_worker();
        }
        return std::make_unique<rapidsmp::ucxx::InitializedRank>(shared_resources);
    } else {
        if (worker == nullptr) {
            worker = create_worker();
        }
        auto shared_resources =
            std::make_shared<rapidsmp::ucxx::SharedResources>(worker, true, nranks);

        // Create listener
        shared_resources->register_listener(
            worker->createListener(0, listener_callback, shared_resources.get())
        );
        auto listener = shared_resources->get_listener();

        // TODO: Enable when Logger can be created before the UCXX communicator object.
        // See https://github.com/rapidsai/rapids-multi-gpu/issues/65 .
        // log.info("Root running at address ", listener->getIp(), ":",
        // listener->getPort());

        auto control_callback = ::ucxx::AmReceiverCallbackType(
            [shared_resources](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
                control_unpack(req->getRecvBuffer(), ep, shared_resources);
            }
        );

        worker->registerAmReceiverCallback(
            shared_resources->get_control_callback_info(), control_callback
        );

        return std::make_unique<rapidsmp::ucxx::InitializedRank>(shared_resources);
    }
}

UCXX::UCXX(std::unique_ptr<InitializedRank> ucxx_initialized_rank)
    : shared_resources_(ucxx_initialized_rank->shared_resources_), logger_(this) {
    shared_resources_->logger = &logger_;
}

[[nodiscard]] Rank UCXX::rank() const {
    return shared_resources_->rank();
}

[[nodiscard]] int UCXX::nranks() const {
    return shared_resources_->nranks();
}

constexpr ::ucxx::Tag tag_with_rank(Rank rank, int tag) {
    // The rapidsmp::ucxx::Communicator API uses 32-bit `int` for user tags to match
    // MPI's standard. We can thus pack the rank in the higher 32-bit of UCX's
    // 64-bit tags as aid in identifying the sender of a message. Since we're
    // currently limited to 26-bits for ranks (see
    // `rapidsmp::ucxx::shuffler::Shuffler::get_new_cid()`), we are essentially using
    // 58-bits for the tags and the remaining 6-bits may be used in the future,
    // such as to identify groups.
    return ::ucxx::Tag(static_cast<uint64_t>(rank) << 32 | tag);
}

constexpr ::ucxx::TagMask UserTagMask{std::numeric_limits<uint32_t>::max()};

std::shared_ptr<::ucxx::Endpoint> UCXX::get_endpoint(Rank rank) {
    Logger& log = logger();
    try {
        log.trace("Endpoint for rank ", rank, " already available, returning to caller");
        return shared_resources_->get_endpoint(rank);
    } catch (std::out_of_range const&) {
        log.trace(
            "Endpoint for rank ", rank, " not available, requesting listener address"
        );
        auto packed_listener_address_rank =
            control_pack(ControlMessage::QueryListenerAddress, rank);

        auto root_endpoint = get_endpoint(Rank(0));

        auto listener_address_req = root_endpoint->amSend(
            packed_listener_address_rank->data(),
            packed_listener_address_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            shared_resources_->get_control_callback_info()
        );

        while (!listener_address_req->isCompleted()) {
            progress_worker();
        }
        while (true) {
            try {
                auto listener_address = shared_resources_->get_listener_address(rank);
                break;
            } catch (std::out_of_range const&) {
            }

            progress_worker();
        }

        auto listener_address = shared_resources_->get_listener_address(rank);
        auto endpoint = std::visit(
            [this](auto&& remote_address) {
                using T = std::decay_t<decltype(remote_address)>;
                if constexpr (std::is_same_v<T, HostPortPair>) {
                    return shared_resources_->get_worker()->createEndpointFromHostname(
                        remote_address.first, remote_address.second, true
                    );
                } else if constexpr (std::is_same_v<T, std::shared_ptr<::ucxx::Address>>)
                {
                    return shared_resources_->get_worker()
                        ->createEndpointFromWorkerAddress(remote_address, true);
                } else {
                    RAPIDSMP_EXPECTS(false, "Unknown argument type");
                }
            },
            listener_address.address
        );
        shared_resources_->register_endpoint(rank, endpoint);
        auto packed_register_rank = control_pack(ControlMessage::RegisterRank, rank);
        auto register_rank_req = endpoint->amSend(
            packed_register_rank->data(),
            packed_register_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            shared_resources_->get_control_callback_info()
        );
        while (!register_rank_req->isCompleted()) {
            progress_worker();
        }

        log.trace(
            "Endpoint for rank ",
            rank,
            " established successfully, requesting listener address"
        );

        return endpoint;
    }
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<std::vector<uint8_t>> msg,
    Rank rank,
    Tag tag,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    auto req = get_endpoint(rank)->tagSend(
        msg->data(),
        msg->size(),
        tag_with_rank(shared_resources_->rank(), static_cast<int>(tag))
    );
    return std::make_unique<Future>(req, br->move(std::move(msg), stream));
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<Buffer> msg, Rank rank, Tag tag, rmm::cuda_stream_view stream
) {
    auto req = get_endpoint(rank)->tagSend(
        msg->data(), msg->size, tag_with_rank(shared_resources_->rank(), tag)
    );
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> UCXX::recv(
    Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer, rmm::cuda_stream_view stream
) {
    auto req = get_endpoint(rank)->tagRecv(
        recv_buffer->data(),
        recv_buffer->size,
        tag_with_rank(rank, tag),
        ::ucxx::TagMaskFull
    );
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> UCXX::recv_any(Tag tag) {
    Logger& log = logger();
    auto probe = shared_resources_->get_worker()->tagProbe(
        ::ucxx::Tag(static_cast<int>(tag)), UserTagMask
    );
    auto msg_available = probe.first;
    auto info = probe.second;
    auto sender_rank = static_cast<Rank>(info.senderTag >> 32);
    if (!msg_available) {
        return {nullptr, 0};
    }
    auto msg = std::make_unique<std::vector<uint8_t>>(info.length
    );  // TODO: choose between host and device

    auto req = shared_resources_->get_worker()->tagRecv(
        msg->data(), msg->size(), ::ucxx::Tag(static_cast<int>(tag)), UserTagMask
    );

    while (!req->isCompleted()) {
        log.warn(
            "block-receiving a messager larger than the normal ",
            "eager threshold (",
            msg->size(),
            " bytes)"
        );
        progress_worker();
    }

    return {std::move(msg), sender_rank};
}

std::vector<std::size_t> UCXX::test_some(
    std::vector<std::unique_ptr<Communicator::Future>> const& future_vector
) {
    progress_worker();
    std::vector<size_t> completed;
    for (size_t i = 0; i < future_vector.size(); i++) {
        auto ucxx_future = dynamic_cast<Future const*>(future_vector[i].get());
        RAPIDSMP_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
        if (ucxx_future->req_->isCompleted()) {
            completed.push_back(i);
        }
    }
    return completed;
}

std::vector<std::size_t> UCXX::test_some(
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
        future_map
) {
    progress_worker();
    std::vector<size_t> completed;
    for (auto const& [key, future] : future_map) {
        auto ucxx_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMP_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
        if (ucxx_future->req_->isCompleted()) {
            completed.push_back(key);
        }
    }
    return completed;
}

void UCXX::barrier() {
    Logger& log = logger();
    log.trace("Barrier started on rank ", shared_resources_->rank());
    shared_resources_->barrier();
    log.trace("Barrier completed on rank ", shared_resources_->rank());
}

std::unique_ptr<Buffer> UCXX::get_gpu_data(std::unique_ptr<Communicator::Future> future) {
    auto ucxx_future = dynamic_cast<Future*>(future.get());
    RAPIDSMP_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
    RAPIDSMP_EXPECTS(ucxx_future->data_ != nullptr, "future has no data");
    return std::move(ucxx_future->data_);
}

std::string UCXX::str() const {
    unsigned major, minor, release;
    ucp_get_version(&major, &minor, &release);

    std::stringstream ss;
    ss << "UCXX(rank=" << shared_resources_->rank()
       << ", nranks: " << shared_resources_->nranks() << ", ucx-version=" << major << "."
       << minor << "." << release << ")";
    return ss.str();
}

UCXX::~UCXX() noexcept {
    Logger& log = logger();
    log.trace("UCXX destructor");
    shared_resources_->get_worker()->stopProgressThread();
    shared_resources_->logger = nullptr;
}

void UCXX::progress_worker() {
    shared_resources_->progress_worker();
}

ListenerAddress UCXX::listener_address() {
    return shared_resources_->get_listener_address(shared_resources_->rank());
}

}  // namespace ucxx

}  // namespace rapidsmp
