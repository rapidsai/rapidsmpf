/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>

#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

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
    ReplyListenerAddress  ///< Reply to `QueryListenerAddress` with the listener address
};

enum class ListenerAddressType {
    WorkerAddress = 0,
    HostPort,
    Undefined
};

using Rank = rapidsmpf::Rank;
using HostPortPair = rapidsmpf::ucxx::HostPortPair;
using RemoteAddress = rapidsmpf::ucxx::RemoteAddress;
using ListenerAddress = rapidsmpf::ucxx::ListenerAddress;
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
        : req_{std::move(req)}, data_buffer_{std::move(data)} {}

    ~HostFuture() noexcept = default;

    HostFuture(HostFuture&&) = default;  ///< Movable.
    HostFuture(HostFuture&) = delete;  ///< Not copyable.

    [[nodiscard]] bool completed() const {
        return req_->isCompleted();
    }

  private:
    std::shared_ptr<::ucxx::Request>
        req_;  ///< The UCXX request associated with the operation.
    std::unique_ptr<std::vector<uint8_t>> data_buffer_;  ///< The data buffer.
};

}  // namespace

class SharedResources {
  private:
    std::shared_ptr<::ucxx::Worker> worker_{nullptr};  ///< UCXX Listener
    std::shared_ptr<::ucxx::Listener> listener_{nullptr};  ///< UCXX Listener
    Rank rank_{Rank(-1)};  ///< Rank of the current process
    Rank nranks_{0};  ///< Number of ranks in the communicator
    std::atomic<Rank> next_rank_{
        1
    };  ///< Rank to assign for the next client that connects (root only)
    EndpointsMap endpoints_{};  ///< Map of UCP handle to UCXX endpoints of known ranks
    RankToEndpointMap rank_to_endpoint_{};  ///< Map of ranks to UCXX endpoints
    RankToListenerAddressMap
        rank_to_listener_address_{};  ///< Map of rank to listener addresses
    const ::ucxx::AmReceiverCallbackInfo control_callback_info_{
        "rapidsmpf", 0
    };  ///< UCXX callback info for control messages
    std::vector<std::unique_ptr<HostFuture>>
        futures_{};  ///< Futures to incomplete requests.
    std::vector<std::function<void()>>
        delayed_progress_callbacks_{};  ///< Callbacks from incomplete requests to execute
                                        ///< before progressing the worker
    std::mutex endpoints_mutex_{};  ///< Mutex to control access to `endpoints_`
    std::mutex futures_mutex_{};  ///< Mutex to control access to `futures_`
    std::mutex listener_mutex_{};  ///< Mutex to control access to `listener_` and
                                   ///< `rank_to_listener_address_`
    std::mutex delayed_progress_callbacks_mutex_{};  ///< Mutex to control access to
                                                     ///< `delayed_progress_callbacks_`
    bool endpoint_error_handling_{
        false
    };  ///< Whether to request UCX endpoint error handling. This is currently disabled
        ///< as it impacts performance very negatively.
        ///< See https://github.com/rapidsai/rapidsmpf/issues/140.
    std::atomic<std::uint64_t> progress_count{
        0
    };  ///< Counts how many times `maybe_progress_worker` has been called

  public:
    UCXX::Logger* logger{nullptr};  ///< UCXX logger

    /**
     * @brief Construct UCXX shared resources.
     *
     * Construct UCXX shared resources, assigning the proper rank 0 for root,
     * other ranks must call `set_rank()` at the appropriate time.
     *
     * @param worker The UCXX worker, or nullptr to create one internally.
     * @param root Whether the rank is the root rank.
     * @param nranks The number of ranks requested for the cluster.
     */
    SharedResources(std::shared_ptr<::ucxx::Worker> worker, bool root, Rank nranks)
        : worker_{std::move(worker)}, rank_{Rank(root ? 0 : -1)}, nranks_{nranks} {}

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
    [[nodiscard]] Rank nranks() const {
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
        RAPIDSMPF_EXPECTS(rank_ == 0, "This method can only be called by rank 0");
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

    /**
     * @brief Gets the UCXX context.
     *
     * Returns the UCXX context from the worker.
     *
     * @return The UCXX context.
     * @throws std::logic_error if the context cannot be obtained from the worker.
     */
    [[nodiscard]] std::shared_ptr<::ucxx::Context> get_context() {
        auto context = std::dynamic_pointer_cast<::ucxx::Context>(worker_->getParent());
        RAPIDSMPF_EXPECTS(context != nullptr, "Failed to get UCXX context from worker");
        return context;
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
        RAPIDSMPF_EXPECTS(
            rank_to_listener_address_
                .emplace(
                    rank_, ListenerAddress{.address = worker->getAddress(), .rank = rank_}
                )
                .second,
            "listener for given rank already exists"
        );
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
        RAPIDSMPF_EXPECTS(
            rank_to_endpoint_.emplace(rank, endpoint).second,
            "endpoint for given rank already exists"
        );
        RAPIDSMPF_EXPECTS(
            endpoints_.emplace(endpoint->getHandle(), std::move(endpoint)).second,
            "endpoint handle already exists"
        );
    }

    /**
     * @brief Registers an endpoint without a rank.
     *
     * Registers an endpoint to a remote process with a still unknown rank.
     *
     * @param endpoint The endpoint to register.
     */
    void register_endpoint(std::shared_ptr<::ucxx::Endpoint> endpoint) {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        RAPIDSMPF_EXPECTS(
            endpoints_.emplace(endpoint->getHandle(), std::move(endpoint)).second,
            "endpoint handle already exists"
        );
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
    [[nodiscard]] std::shared_ptr<::ucxx::Endpoint> get_endpoint(
        ucp_ep_h const ep_handle
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
        RAPIDSMPF_EXPECTS(
            rank_to_listener_address_.emplace(rank, std::move(listener_address)).second,
            "listener for given rank already exists"
        );
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
        while (rank_ == 0 && rank_to_endpoint_.size() != static_cast<size_t>(nranks())) {
            progress_worker();
        }

        if (rank_ == 0) {
            std::vector<std::shared_ptr<::ucxx::Request>> requests;
            requests.reserve(static_cast<size_t>(nranks() - 1));
            // send to all other ranks
            for (auto& [rank, endpoint] : rank_to_endpoint_) {
                if (rank == 0) {
                    continue;
                }
                requests.push_back(endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST));
            }
            while (std::ranges::any_of(requests, [](auto const& req) {
                return !req->isCompleted();
            }))
            {
                progress_worker();
            }
            requests.clear();

            // receive from all other ranks
            for (auto& [rank, endpoint] : rank_to_endpoint_) {
                if (rank == 0) {
                    continue;
                }
                requests.push_back(endpoint->amRecv());
            }
            while (std::ranges::any_of(requests, [](auto const& req) {
                return !req->isCompleted();
            }))
            {
                progress_worker();
            }
        } else {  // non-root ranks respond to root's broadcast
            auto endpoint = get_endpoint(Rank(0));

            auto req = endpoint->amRecv();
            while (!req->isCompleted()) {
                progress_worker();
            }

            req = endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST);
            while (!req->isCompleted()) {
                progress_worker();
            }
        }
    }

    void clear_completed_futures() {
        std::lock_guard<std::mutex> lock(futures_mutex_);
        std::erase_if(futures_, [](std::unique_ptr<HostFuture> const& element) {
            return element->completed();
        });
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
            // TODO: Support blocking progress mode in addition to polling
            worker_->progress();
        }

        clear_completed_futures();
    }

    void maybe_progress_worker() {
        // The value here is the default borrowed from OpenMPI:
        // https://github.com/open-mpi/ompi/blob/7ad7adad676773fc61203e9a536d17b2ebdfa9c8/opal/mca/common/ucx/common_ucx.c#L42
        if (++progress_count % 100) {
            progress_worker();
        }
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
        delayed_progress_callbacks_.emplace_back(std::move(callback));
    }

    /**
     * @brief Whether endpoint error handling should be enabled.
     *
     * @return `true` if endpoint error handling should be enabled, `false` otherwise.
     */
    bool endpoint_error_handling() {
        return endpoint_error_handling_;
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
        overloaded{
            [&listener_address](HostPortPair const& remote_address) {
                size_t offset{0};
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
            },
            [&listener_address](std::shared_ptr<::ucxx::Address> const& remote_address) {
                size_t offset{0};
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
        RAPIDSMPF_EXPECTS(false, "Wrong type");
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

    if (control == ControlMessage::AssignRank
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
    } else if (control == ControlMessage::QueryRank
               || control == ControlMessage::ReplyListenerAddress)
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
        RAPIDSMPF_EXPECTS(false, "Invalid control type");
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
 * 3. ReplyListenerAddress: Handle reply from root rank, associating the
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
    std::shared_ptr<rapidsmpf::ucxx::SharedResources> shared_resources
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
                    worker_address, shared_resources->endpoint_error_handling()
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
 * which rank it should now respond to in the communicator world.
 */
void listener_callback(ucp_conn_request_h conn_request, void* arg) {
    auto shared_resources = reinterpret_cast<rapidsmpf::ucxx::SharedResources*>(arg);

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
        conn_request, shared_resources->endpoint_error_handling()
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
void create_cuda_context_callback(void* /* callbackArg */) {
    cudaFree(nullptr);
}

}  // namespace

InitializedRank::InitializedRank(
    std::shared_ptr<rapidsmpf::ucxx::SharedResources> shared_resources
)
    : shared_resources_(std::move(shared_resources)) {}

std::unique_ptr<rapidsmpf::ucxx::InitializedRank> init(
    std::shared_ptr<::ucxx::Worker> worker,
    Rank nranks,
    std::optional<RemoteAddress> remote_address,
    config::Options options
) {
    auto progress_mode =
        options.get<ProgressMode>("ucxx_progress_mode", [](auto const& s) {
            if (s.empty()) {
                return ProgressMode::ThreadBlocking;
            } else if (s == "blocking") {
                return ProgressMode::Blocking;
            } else if (s == "polling") {
                return ProgressMode::Polling;
            } else if (s == "thread-blocking") {
                return ProgressMode::ThreadBlocking;
            } else if (s == "thread-polling") {
                return ProgressMode::ThreadPolling;
            } else {
                RAPIDSMPF_FAIL("Invalid progress mode");
            }
        });

    auto create_worker = [progress_mode]() {
        auto context = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
        auto worker = context->createWorker(false);

        RAPIDSMPF_EXPECTS(
            progress_mode != ProgressMode::Blocking,
            "Blocking progress mode not implemented yet."
        );

        if (progress_mode == ProgressMode::ThreadBlocking
            || progress_mode == ProgressMode::ThreadPolling)
        {
            worker->setProgressThreadStartCallback(create_cuda_context_callback, nullptr);
            worker->startProgressThread(progress_mode == ProgressMode::ThreadPolling);
        };

        return worker;
    };

    auto register_self_endpoint = [](::ucxx::Worker& worker,
                                     rapidsmpf::ucxx::SharedResources& shared_resources) {
        auto self_ep = worker.createEndpointFromWorkerAddress(
            worker.getAddress(), shared_resources.endpoint_error_handling()
        );
        shared_resources.register_endpoint(shared_resources.rank(), std::move(self_ep));
    };

    if (remote_address) {
        if (worker == nullptr) {
            worker = create_worker();
        }
        auto shared_resources =
            std::make_shared<rapidsmpf::ucxx::SharedResources>(worker, false, nranks);

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
        // See https://github.com/rapidsai/rapidsmpf/issues/65 .
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
            overloaded{
                [shared_resources](HostPortPair const& remote_address) {
                    return shared_resources->get_worker()->createEndpointFromHostname(
                        remote_address.first,
                        remote_address.second,
                        shared_resources->endpoint_error_handling()
                    );
                },
                [shared_resources](
                    std::shared_ptr<::ucxx::Address> const& remote_address
                ) {
                    auto root_endpoint =
                        shared_resources->get_worker()->createEndpointFromWorkerAddress(
                            remote_address, shared_resources->endpoint_error_handling()
                        );

                    auto packed_listener_address_rank = control_pack(
                        ControlMessage::QueryRank,
                        ListenerAddress{
                            .address = shared_resources->get_worker()->getAddress(),
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
                }
            },
            *remote_address
        );
        shared_resources->register_endpoint(Rank(0), endpoint);

        // Get my rank
        while (shared_resources->rank() == Rank(-1)) {
            shared_resources->progress_worker();
        }

        register_self_endpoint(*worker, *shared_resources);

        // TODO: Enable when Logger can be created before the UCXX communicator object.
        // See https://github.com/rapidsai/rapidsmpf/issues/65 .
        // log.debug("Assigned rank: ", shared_resources->rank());

        if (const HostPortPair* host_port_pair =
                std::get_if<HostPortPair>(&*remote_address))
        {
            RAPIDSMPF_EXPECTS(host_port_pair != nullptr, "Invalid pointer");

            // Inform listener address
            ListenerAddress listener_address = ListenerAddress{
                .address = std::make_pair(listener->getIp(), listener->getPort()),
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
        return std::make_unique<rapidsmpf::ucxx::InitializedRank>(shared_resources);
    } else {
        if (worker == nullptr) {
            worker = create_worker();
        }
        auto shared_resources =
            std::make_shared<rapidsmpf::ucxx::SharedResources>(worker, true, nranks);

        // Create listener
        shared_resources->register_listener(
            worker->createListener(0, listener_callback, shared_resources.get())
        );
        auto listener = shared_resources->get_listener();

        // TODO: Enable when Logger can be created before the UCXX communicator object.
        // See https://github.com/rapidsai/rapidsmpf/issues/65 .
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

        register_self_endpoint(*worker, *shared_resources);

        return std::make_unique<rapidsmpf::ucxx::InitializedRank>(
            std::move(shared_resources)
        );
    }
}

UCXX::UCXX(
    std::unique_ptr<InitializedRank> ucxx_initialized_rank, config::Options options
)
    : shared_resources_(ucxx_initialized_rank->shared_resources_),
      options_{std::move(options)},
      logger_(this, options_) {
    shared_resources_->logger = &logger_;
}

[[nodiscard]] Rank UCXX::rank() const {
    return shared_resources_->rank();
}

[[nodiscard]] Rank UCXX::nranks() const {
    return shared_resources_->nranks();
}

constexpr ::ucxx::Tag tag_with_rank(Rank rank, int tag) {
    // The rapidsmpf::ucxx::Communicator API uses 32-bit `int` for user tags to match
    // MPI's standard. We can thus pack the rank in the higher 32-bit of UCX's
    // 64-bit tags as aid in identifying the sender of a message. Since we're
    // currently limited to 26-bits for ranks (see
    // `rapidsmpf::ucxx::shuffler::Shuffler::get_new_cid()`), we are essentially using
    // 58-bits for the tags and the remaining 6-bits may be used in the future,
    // such as to identify groups.
    return ::ucxx::Tag(static_cast<uint64_t>(rank) << 32 | static_cast<uint64_t>(tag));
}

constexpr ::ucxx::TagMask UserTagMask{std::numeric_limits<uint32_t>::max()};

std::shared_ptr<::ucxx::Endpoint> UCXX::get_endpoint(Rank rank) {
    Logger& log = logger();
    try {
        auto ep = shared_resources_->get_endpoint(rank);
        log.trace("Endpoint for rank ", rank, " already available, returning to caller");
        return ep;
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
            overloaded{
                [this](HostPortPair const& remote_address) {
                    return shared_resources_->get_worker()->createEndpointFromHostname(
                        remote_address.first,
                        remote_address.second,
                        shared_resources_->endpoint_error_handling()
                    );
                },
                [this](std::shared_ptr<::ucxx::Address> const& remote_address) {
                    return shared_resources_->get_worker()
                        ->createEndpointFromWorkerAddress(
                            remote_address, shared_resources_->endpoint_error_handling()
                        );
                }
            },
            listener_address.address
        );
        shared_resources_->register_endpoint(rank, endpoint);

        log.trace(
            "Endpoint for rank ",
            rank,
            " established successfully, requesting listener address"
        );

        return endpoint;
    }
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<std::vector<uint8_t>> msg, Rank rank, Tag tag
) {
    auto req = get_endpoint(rank)->tagSend(
        msg->data(),
        msg->size(),
        tag_with_rank(shared_resources_->rank(), static_cast<int>(tag))
    );
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<Buffer> msg, Rank rank, Tag tag
) {
    RAPIDSMPF_EXPECTS(msg->is_latest_write_done(), "msg must be ready");
    auto req = get_endpoint(rank)->tagSend(
        msg->data(), msg->size, tag_with_rank(shared_resources_->rank(), tag)
    );
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> UCXX::recv(
    Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
) {
    RAPIDSMPF_EXPECTS(recv_buffer != nullptr, "recv buffer is nullptr");
    RAPIDSMPF_EXPECTS(recv_buffer->is_latest_write_done(), "msg must be ready");
    auto req = get_endpoint(rank)->tagRecv(
        recv_buffer->exclusive_data_access(),
        recv_buffer->size,
        tag_with_rank(rank, tag),
        ::ucxx::TagMaskFull
    );
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::unique_ptr<Communicator::Future> UCXX::recv_sync_host_data(
    Rank rank, Tag tag, std::unique_ptr<std::vector<uint8_t>> synced_buffer
) {
    RAPIDSMPF_EXPECTS(synced_buffer != nullptr, "recv host buffer is nullptr");
    auto req = get_endpoint(rank)->tagRecv(
        synced_buffer->data(),
        synced_buffer->size(),
        tag_with_rank(rank, tag),
        ::ucxx::TagMaskFull
    );
    return std::make_unique<Future>(req, std::move(synced_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> UCXX::recv_any(Tag tag) {
    progress_worker();
    auto probe = shared_resources_->get_worker()->tagProbe(
        ::ucxx::Tag(static_cast<int>(tag)), UserTagMask, true
    );
    auto msg_available = probe->isMatched();
    if (!msg_available) {
        return {nullptr, 0};
    }
    auto info = probe->getInfo();
    auto sender_rank = static_cast<Rank>(info.senderTag >> 32);
    auto msg = std::make_unique<std::vector<uint8_t>>(
        info.length
    );  // TODO: choose between host and device

    auto req = shared_resources_->get_worker()->tagRecvWithHandle(msg->data(), probe);

    while (!req->isCompleted()) {
        progress_worker();
    }
    req->checkError();

    return {std::move(msg), sender_rank};
}

std::unique_ptr<std::vector<uint8_t>> UCXX::recv_from(Rank src, Tag tag) {
    progress_worker();
    auto probe = shared_resources_->get_worker()->tagProbe(
        tag_with_rank(src, static_cast<int>(tag)), ::ucxx::TagMaskFull, true
    );
    auto msg_available = probe->isMatched();
    if (!msg_available) {
        return nullptr;
    }
    auto info = probe->getInfo();
    auto msg = std::make_unique<std::vector<uint8_t>>(
        info.length
    );  // TODO: choose between host and device

    auto req = shared_resources_->get_worker()->tagRecvWithHandle(msg->data(), probe);

    while (!req->isCompleted()) {
        progress_worker();
    }
    req->checkError();

    return msg;
}

std::pair<std::vector<std::unique_ptr<Communicator::Future>>, std::vector<std::size_t>>
UCXX::test_some(std::vector<std::unique_ptr<Communicator::Future>>& future_vector) {
    if (future_vector.empty()) {
        return {};
    }
    progress_worker();
    std::vector<size_t> indices;
    indices.reserve(future_vector.size());
    for (size_t i = 0; i < future_vector.size(); i++) {
        auto ucxx_future = dynamic_cast<Future const*>(future_vector[i].get());
        RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
        if (ucxx_future->req_->isCompleted()) {
            ucxx_future->req_->checkError();
            indices.push_back(i);
        } else {
            // We rely on this API returning completed futures in order,
            // since we send acks and then post receives for data
            // buffers in order. UCX completes message in order, but
            // since there is a background progress thread, it might be
            // that we observe req[i]->isCompleted() as false, then
            // req[i+1]->isCompleted() as true (but then
            // req[i]->isCompleted() also would return true, but we
            // don't go back and check).
            // Hence if we observe a "gap" in the completed requests
            // from a rank, we must stop processing to ensure we respond
            // to the ready for data messages in order.
            break;
        }
    }
    if (indices.size() == 0) {
        return {};
    }
    std::vector<std::unique_ptr<Communicator::Future>> completed;
    completed.reserve(indices.size());
    std::ranges::transform(indices, std::back_inserter(completed), [&](std::size_t i) {
        return std::move(future_vector[i]);
    });
    std::erase(future_vector, nullptr);
    return {std::move(completed), std::move(indices)};
}

std::vector<std::size_t> UCXX::test_some(
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
        future_map
) {
    progress_worker();
    std::vector<size_t> completed;
    for (auto const& [key, future] : future_map) {
        auto ucxx_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
        if (ucxx_future->req_->isCompleted()) {
            ucxx_future->req_->checkError();
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

std::unique_ptr<Buffer> UCXX::wait(std::unique_ptr<Communicator::Future> future) {
    auto ucxx_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
    while (!ucxx_future->req_->isCompleted()) {
        progress_worker();
    }
    ucxx_future->req_->checkError();
    ucxx_future->data_buffer_->unlock();
    return std::move(ucxx_future->data_buffer_);
}

std::unique_ptr<Buffer> UCXX::release_data(std::unique_ptr<Communicator::Future> future) {
    auto ucxx_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
    RAPIDSMPF_EXPECTS(ucxx_future->data_buffer_ != nullptr, "future has no data");
    ucxx_future->data_buffer_->unlock();
    return std::move(ucxx_future->data_buffer_);
}

std::unique_ptr<std::vector<uint8_t>> UCXX::release_sync_host_data(
    std::unique_ptr<Communicator::Future> future
) {
    auto ucxx_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
    RAPIDSMPF_EXPECTS(ucxx_future->synced_host_data_ != nullptr, "future has no data");
    return std::move(ucxx_future->synced_host_data_);
}

std::string UCXX::str() const {
    unsigned major, minor, release;
    ucp_get_version(&major, &minor, &release);

    std::stringstream ss;
    ss << "UCXX(rank=" << shared_resources_->rank()
       << ", nranks=" << shared_resources_->nranks() << ", ucx-version=" << major << "."
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
    shared_resources_->maybe_progress_worker();
}

ListenerAddress UCXX::listener_address() {
    return shared_resources_->get_listener_address(shared_resources_->rank());
}

std::shared_ptr<UCXX> UCXX::split() {
    Logger& log = logger();
    log.trace("Splitting communicator on rank ", shared_resources_->rank());

    // Get the context from shared resources
    auto context = shared_resources_->get_context();

    // Create a new worker using the same context
    auto worker = context->createWorker(false);
    worker->setProgressThreadStartCallback(create_cuda_context_callback, nullptr);
    worker->startProgressThread(true);

    // Create new shared resources with nranks=1
    auto shared_resources = std::make_shared<SharedResources>(worker, true, 1);

    // Create listener
    shared_resources->register_listener(
        worker->createListener(0, listener_callback, shared_resources.get())
    );

    // Set up control callback
    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [shared_resources](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            control_unpack(req->getRecvBuffer(), ep, shared_resources);
        }
    );

    worker->registerAmReceiverCallback(
        shared_resources->get_control_callback_info(), control_callback
    );

    // Create the new UCXX instance
    auto initialized_rank = std::make_unique<InitializedRank>(shared_resources);
    return std::make_shared<UCXX>(std::move(initialized_rank), options_);
}

}  // namespace ucxx

}  // namespace rapidsmpf
