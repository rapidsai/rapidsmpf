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

#include <array>
#include <mutex>
#include <utility>

#include <rapidsmp/communicator/ucxx.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

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
 * 3. RegisterRank: After connecting to another (non-root) rank with that rank's
 * listener address, inform own rank so that the remote rank can register the
 * recently-established endpoint as corresponding to this rank.
 * 4. ReplyListenerAddress: Root rank reply to `QueryListenerAddress` request
 * containing the listener address of the rank being requested.
 */
enum class ControlMessage {
    AssignRank = 0,  //< Root assigns a rank to incoming client connection
    QueryListenerAddress,  ///< Ask for the remote endpoint's listener address
    RegisterRank,  ///< Inform rank to remote process (non-root) after endpoint is
                   ///< established
    ReplyListenerAddress  ///< Reply to `QueryListenerAddress` with the listener address
};

using Rank = rapidsmp::Rank;
using ListenerAddress = rapidsmp::ListenerAddress;
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

class UCXXSharedResources {
  public:
    std::shared_ptr<::ucxx::Listener> listener_{nullptr};  ///< UCXX Listener
    Rank rank_{Rank(-1)};  ///< Rank of the current process
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
        std::vector<std::unique_ptr<HostFuture>>()
    };  ///< Futures to incomplete requests.
    std::mutex endpoints_mutex_{};
    std::mutex futures_mutex_{};
    std::mutex listener_mutex_{};

    /**
     * @brief Construct UCXX shared resources.
     *
     * Constructor UCXX shared resources, assigning the proper rank 0 for root,
     * other ranks must call `set_rank()` at the appropriate time.
     *
     * @param root Whether the rank is the root rank.
     * @param nranks The number of ranks requested for the cluster.
     */
    UCXXSharedResources(bool root) : rank_(Rank(root ? 0 : -1)) {}

    UCXXSharedResources(UCXXSharedResources&&) = delete;  ///< Not movable.
    UCXXSharedResources(UCXXSharedResources&) = delete;  ///< Not copyable.

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
     * @brief Gets the next worker rank.
     *
     * Returns the next available worker rank. This method can only be called
     * by root rank (rank 0).
     *
     * @return The next available worker rank.
     * @throws std::runtime_error If called by rank other than 0.
     */
    Rank get_next_worker_rank() {
        if (rank_ != 0)
            throw std::runtime_error("This method can only be called by rank 0");
        return next_rank_++;
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
        listener_ = listener;
        auto listener_address = ListenerAddress{
            .host = listener->getIp(), .port = listener->getPort(), .rank = rank_
        };
        rank_to_listener_address_[rank_] = listener_address;
    }

    /**
     * @brief Registers an endpoint for a specific rank.
     *
     * Registers an endpoint and associate it with a specific rank.
     *
     * @param rank The rank to register the endpoint for.
     * @param endpoint The endpoint to register.
     */
    void register_endpoint(const Rank rank, std::shared_ptr<::ucxx::Endpoint> endpoint) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        rank_to_endpoint_[rank] = endpoint;
        endpoints_[endpoint->getHandle()] = endpoint;
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
        endpoints_[endpoint->getHandle()] = endpoint;
    }

    /**
     * @brief Associate endpoint with a specific rank.
     *
     * Associate a previously registered endpoint to a specific rank.
     *
     * @param rank The rank to register the endpoint for.
     * @param endpoint_handle The handle of the endpoint to register.
     */
    void associate_endpoint_rank(const Rank rank, const ucp_ep_h endpoint_handle) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        auto endpoint = endpoints_[endpoint_handle];
        rank_to_endpoint_[rank] = endpoint;
    }

    /**
     * @brief Gets the listener.
     *
     * Returns the registered listener.
     *
     * @return The registered listener.
     */
    std::shared_ptr<::ucxx::Listener> get_listener() {
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
    std::shared_ptr<::ucxx::Endpoint> get_endpoint(const ucp_ep_h ep_handle) {
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
    std::shared_ptr<::ucxx::Endpoint> get_endpoint(const Rank rank) {
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
    ListenerAddress get_listener_address(const Rank rank) {
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
    void register_listener_address(
        const Rank rank, const ListenerAddress listener_address
    ) {
        std::lock_guard<std::mutex> lock(listener_mutex_);
        rank_to_listener_address_[rank] = listener_address;
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

    void clear_completed_futures() {
        std::lock_guard<std::mutex> lock(futures_mutex_);
        futures_.erase(
            std::remove_if(
                futures_.begin(),
                futures_.end(),
                [](const std::unique_ptr<HostFuture>& element) {
                    return element->completed();
                }
            ),
            futures_.end()
        );
    }
};

namespace {

size_t get_size(const ControlData& data) {
    return std::visit(
        [](const auto& data) { return sizeof(std::decay_t<decltype(data)>); }, data
    );
}

void encode(void* dest, void const* src, size_t bytes, size_t& offset) {
    memcpy(static_cast<char*>(dest) + offset, src, bytes);
    offset += bytes;
}

void decode(void* dest, const void* src, size_t bytes, size_t& offset) {
    memcpy(dest, static_cast<const char*>(src) + offset, bytes);
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
    const ListenerAddress& listener_address
) {
    size_t offset{0};
    size_t host_size = listener_address.host.size();
    const size_t total_size = sizeof(host_size) + host_size
                              + sizeof(listener_address.port)
                              + sizeof(listener_address.rank);
    auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

    auto encode_ = [&offset, &packed](const void* data, size_t bytes) {
        encode(packed->data(), data, bytes, offset);
    };

    encode_(&host_size, sizeof(host_size));
    encode_(listener_address.host.data(), host_size);
    encode_(&listener_address.port, sizeof(listener_address.port));
    encode_(&listener_address.rank, sizeof(listener_address.rank));

    return packed;
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

    size_t host_size;
    decode_(&host_size, sizeof(size_t));

    ListenerAddress listener_address;
    listener_address.host.resize(host_size);
    decode_(listener_address.host.data(), host_size);

    decode_(&listener_address.port, sizeof(listener_address.port));
    decode_(&listener_address.rank, sizeof(listener_address.rank));

    return listener_address;
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
    const size_t total_size = sizeof(control) + get_size(data);

    auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

    auto encode_ = [&offset, &packed](void const* data, size_t bytes) {
        encode(packed->data(), data, bytes, offset);
    };

    encode_(&control, sizeof(control));
    if (control == ControlMessage::AssignRank || control == ControlMessage::RegisterRank
        || control == ControlMessage::QueryListenerAddress)
    {
        auto rank = std::get<Rank>(data);
        encode_(&rank, sizeof(rank));
    } else if (control == ControlMessage::ReplyListenerAddress) {
        auto listener_address = std::get<ListenerAddress>(data);
        auto packed_listener_address = listener_address_pack(listener_address);
        size_t packed_listener_address_size = packed_listener_address->size();
        encode_(&packed_listener_address_size, sizeof(packed_listener_address_size));
        encode_(packed_listener_address->data(), packed_listener_address_size);
    }

    return packed;
};

/**
 * @brief Unpack control message.
 *
 * Unpack (i.e., deserialize) `ControlMessage` and `ControlData` received over
 * the wire, and appropriately handle contained data.
 *
 * 1. AssignRank: Calls `UCXXSharedResources->set_rank()` to set own rank.
 * 2. QueryListenerAddress: Get the previously-registered listener address for
 * the rank being requested and reply the requester with the listener address.
 * A future with the reply is stored via `UCXXSharedResources->add_future()`.
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
    std::shared_ptr<rapidsmp::UCXXSharedResources> shared_resources
) {
    size_t offset{0};

    auto decode_ = [&offset, &buffer](void* data, size_t bytes) {
        decode(data, buffer->data(), bytes, offset);
    };

    ControlMessage control;
    decode_(&control, sizeof(ControlMessage));

    if (control == ControlMessage::AssignRank) {
        Rank rank;
        decode_(&rank, sizeof(rank));
        shared_resources->set_rank(rank);
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
            listener_address.rank,
            ListenerAddress{
                .host = listener_address.host,
                .port = listener_address.port,
                .rank = listener_address.rank,
            }
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
            shared_resources->control_callback_info_
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
    auto shared_resources = reinterpret_cast<rapidsmp::UCXXSharedResources*>(arg);

    ucp_conn_request_attr_t attr{};
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    auto status = ucp_conn_request_query(conn_request, &attr);
    if (status != UCS_OK) {
        // TODO: Switch to logger
        std::cout << "Failed to create endpoint to client" << std::endl;
        return;
    }

    std::array<char, INET6_ADDRSTRLEN> ip_str;
    std::array<char, INET6_ADDRSTRLEN> port_str;
    ::ucxx::utils::sockaddr_get_ip_port_str(
        &attr.client_address, ip_str.data(), port_str.data(), INET6_ADDRSTRLEN
    );
    // std::cout << "Server received a connection request from client at address "
    //           << ip_str.data() << ":" << port_str.data() << std::endl;

    auto endpoint =
        shared_resources->listener_->createEndpointFromConnRequest(conn_request, true);

    if (shared_resources->rank_ == 0) {
        Rank client_rank = shared_resources->get_next_worker_rank();
        shared_resources->register_endpoint(client_rank, endpoint);
        auto packed_client_rank = control_pack(ControlMessage::AssignRank, client_rank);
        auto req = endpoint->amSend(
            packed_client_rank->data(),
            packed_client_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            shared_resources->control_callback_info_
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

UCXX::UCXX(std::shared_ptr<::ucxx::Worker> worker, std::uint32_t nranks)
    : worker_(std::move(worker)),
      shared_resources_(std::make_shared<rapidsmp::UCXXSharedResources>(true)),
      nranks_(nranks),
      logger_(this) {
    if (worker_ == nullptr) {
        auto context = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
        worker_ = context->createWorker(false);
        // TODO: Allow other modes
        worker_->setProgressThreadStartCallback(create_cuda_context_callback, nullptr);
        worker_->startProgressThread(false);
    }

    // Create listener
    shared_resources_->register_listener(
        worker_->createListener(0, listener_callback, shared_resources_.get())
    );
    auto listener = shared_resources_->get_listener();

    Logger& log = logger();
    log.info("Root running at address ", listener->getIp(), ":", listener->getPort());

    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            control_unpack(req->getRecvBuffer(), ep, shared_resources_);
        }
    );

    worker_->registerAmReceiverCallback(
        shared_resources_->control_callback_info_, control_callback
    );
}

UCXX::UCXX(
    std::shared_ptr<::ucxx::Worker> worker,
    std::uint32_t nranks,
    std::string root_host,
    uint16_t root_port
)
    : worker_(std::move(worker)),
      shared_resources_(std::make_shared<rapidsmp::UCXXSharedResources>(false)),
      nranks_(nranks),
      logger_(this) {
    if (worker_ == nullptr) {
        auto context = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
        worker_ = context->createWorker(false);
        // TODO: Allow other modes
        worker_->setProgressThreadStartCallback(create_cuda_context_callback, nullptr);
        worker_->startProgressThread(true);
    }

    // Create listener
    shared_resources_->register_listener(
        worker_->createListener(0, listener_callback, shared_resources_.get())
    );
    auto listener = shared_resources_->get_listener();

    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            auto buffer = req->getRecvBuffer();
            control_unpack(req->getRecvBuffer(), ep, shared_resources_);
        }
    );

    worker_->registerAmReceiverCallback(
        shared_resources_->control_callback_info_, control_callback
    );

    // Connect to root
    Logger& log = logger();
    log.debug(
        "Connecting to root node at ",
        root_host,
        ":",
        root_port,
        ". Current rank: ",
        shared_resources_->rank_
    );
    auto endpoint = worker_->createEndpointFromHostname(root_host, root_port, true);
    shared_resources_->register_endpoint(Rank(0), endpoint);

    // Get my rank
    while (shared_resources_->rank_ == Rank(-1)) {
        progress_worker();
    }
    log.debug("Assigned rank: ", shared_resources_->rank_);

    // Inform listener address
    ListenerAddress listener_address = ListenerAddress{
        .host = "localhost", .port = listener->getPort(), .rank = shared_resources_->rank_
    };
    auto packed_listener_address =
        control_pack(ControlMessage::ReplyListenerAddress, listener_address);
    auto req = endpoint->amSend(
        packed_listener_address->data(),
        packed_listener_address->size(),
        UCS_MEMORY_TYPE_HOST,
        shared_resources_->control_callback_info_
    );
    while (!req->isCompleted())
        progress_worker();
}

[[nodiscard]] Rank UCXX::rank() const {
    return shared_resources_->rank_;
}

::ucxx::Tag tag_with_rank(Rank rank, int tag) {
    // The rapidsmp::Communicator API uses 32-bit `int` for user tags to match
    // MPI's standard. We can thus pack the rank in the higher 32-bit of UCX's
    // 64-bit tags as aid in identifying the sender of a message. Since we're
    // currently limited to 26-bits for ranks (see
    // `rapidsmp::shuffler::Shuffler::get_new_cid()`), we are essentially using
    // 58-bits for the tags and the remaining 6-bits may be used in the future,
    // such as to identify groups.
    return ::ucxx::Tag(static_cast<uint64_t>(rank) << 32 | tag);
}

constexpr ::ucxx::TagMask UserTagMask{std::numeric_limits<int>::max()};

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
            shared_resources_->control_callback_info_
        );

        while (!listener_address_req->isCompleted()) {
            progress_worker();
        }
        while (true) {
            try {
                shared_resources_->get_listener_address(rank);
                break;
            } catch (std::out_of_range const&) {
            }

            progress_worker();
        }

        auto listener_address = shared_resources_->get_listener_address(rank);
        auto endpoint = worker_->createEndpointFromHostname(
            listener_address.host, listener_address.port, true
        );
        shared_resources_->register_endpoint(rank, endpoint);
        auto packed_register_rank = control_pack(ControlMessage::RegisterRank, rank);
        auto register_rank_req = endpoint->amSend(
            packed_register_rank->data(),
            packed_register_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            shared_resources_->control_callback_info_
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
    int tag,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    auto req = get_endpoint(rank)->tagSend(
        msg->data(), msg->size(), tag_with_rank(shared_resources_->rank_, tag)
    );
    return std::make_unique<Future>(req, br->move(std::move(msg), stream));
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream
) {
    auto req = get_endpoint(rank)->tagSend(
        msg->data(), msg->size, tag_with_rank(shared_resources_->rank_, tag)
    );
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> UCXX::recv(
    Rank rank, int tag, std::unique_ptr<Buffer> recv_buffer, rmm::cuda_stream_view stream
) {
    auto req = get_endpoint(rank)->tagRecv(
        recv_buffer->data(),
        recv_buffer->size,
        tag_with_rank(rank, tag),
        ::ucxx::TagMaskFull
    );
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> UCXX::recv_any(int tag) {
    Logger& log = logger();
    auto probe = worker_->tagProbe(::ucxx::Tag(tag), UserTagMask);
    auto msg_available = probe.first;
    auto info = probe.second;
    auto sender_rank = static_cast<Rank>(info.senderTag >> 32);
    if (!msg_available) {
        return {nullptr, 0};
    }
    auto msg = std::make_unique<std::vector<uint8_t>>(info.length
    );  // TODO: choose between host and device

    auto req = worker_->tagRecv(msg->data(), msg->size(), ::ucxx::Tag(tag), UserTagMask);

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
    log.trace("Barrier started on rank ", shared_resources_->rank_);
    while (shared_resources_->rank_ == 0
           && shared_resources_->rank_to_listener_address_.size()
                  != static_cast<uint64_t>(nranks()))
        progress_worker();

    if (shared_resources_->rank_ == 0) {
        std::vector<std::shared_ptr<::ucxx::Request>> requests;
        for (auto& rank_to_endpoint : shared_resources_->rank_to_endpoint_) {
            auto& endpoint = rank_to_endpoint.second;
            requests.push_back(endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST));
        }
        while (std::any_of(requests.cbegin(), requests.cend(), [](auto const& req) {
            return !req->isCompleted();
        }))
            progress_worker();

        requests.clear();

        for (auto& rank_to_endpoint : shared_resources_->rank_to_endpoint_) {
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
    log.trace("Barrier completed on rank ", shared_resources_->rank_);
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
    ss << "UCXX(rank=" << shared_resources_->rank_ << ", nranks: " << nranks_
       << ", ucx-version=" << major << "." << minor << "." << release << ")";
    return ss.str();
}

UCXX::~UCXX() noexcept {
    Logger& log = logger();
    log.trace("UCXX destructor");
    worker_->stopProgressThread();
}

void UCXX::progress_worker() {
    if (!worker_->isProgressThreadRunning()) {
        worker_->progress();
        // TODO: Support blocking progress mode
    }

    shared_resources_->clear_completed_futures();
}

ListenerAddress UCXX::listener_address() {
    return shared_resources_->get_listener_address(shared_resources_->rank_);
}

}  // namespace rapidsmp
