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

#include "rapidsmp/communicator/ucxx.hpp"

#include <array>
#include <utility>

#include <rapidsmp/error.hpp>

namespace rapidsmp {

namespace ucxx {
// void init(int* argc, char*** argv) {
//     int provided;
//
//     // Initialize MPI with the desired level of thread support
//     MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
//
//     RAPIDSMP_EXPECTS(
//         provided == MPI_THREAD_MULTIPLE,
//         "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
//     );
// }
//
// void detail::check_mpi_error(int error_code, const char* file, int line) {
//     if (error_code != MPI_SUCCESS) {
//         std::array<char, MPI_MAX_ERROR_STRING> error_string;
//         int error_length;
//         MPI_Error_string(error_code, error_string.data(), &error_length);
//         std::cerr << "MPI error at " << file << ":" << line << ": "
//                   << std::string(error_string.data(), error_length) << std::endl;
//         MPI_Abort(MPI_COMM_WORLD, error_code);
//     }
// }
}  // namespace ucxx

// namespace {
// void check_mpi_thread_support() {
//     int level;
//     RAPIDSMP_MPI(MPI_Query_thread(&level));
//
//     std::string level_str;
//     switch (level) {
//     case MPI_THREAD_SINGLE:
//         level_str = "MPI_THREAD_SINGLE";
//         break;
//     case MPI_THREAD_FUNNELED:
//         level_str = "MPI_THREAD_FUNNELED";
//         break;
//     case MPI_THREAD_SERIALIZED:
//         level_str = "MPI_THREAD_SERIALIZED";
//         break;
//     case MPI_THREAD_MULTIPLE:
//         level_str = "MPI_THREAD_MULTIPLE";
//         break;
//     default:
//         throw std::logic_error("MPI_Query_thread(): unknown thread level support");
//     }
//     RAPIDSMP_EXPECTS(
//         level == MPI_THREAD_MULTIPLE,
//         "MPI thread level support " + level_str
//             + " isn't sufficient, need MPI_THREAD_MULTIPLE"
//     );
// }
// }  // namespace

static size_t get_size(const rapidsmp::ControlData& data) {
    return std::visit(
        [](const auto& data) { return sizeof(std::decay_t<decltype(data)>); }, data
    );
}

static void encode_(void* dest, void const* src, size_t bytes, size_t& offset) {
    // std::cout << "Encoding " << bytes << " bytes" << std::endl;
    // for (size_t i = 0; i < bytes; ++i) {
    //     printf("\\0x%02x", static_cast<const unsigned char*>(src)[i]);
    // }
    // std::cout << std::endl;
    memcpy(static_cast<char*>(dest) + offset, src, bytes);
    offset += bytes;
}

static void decode_(void* dest, const void* src, size_t bytes, size_t& offset) {
    memcpy(dest, static_cast<const char*>(src) + offset, bytes);
    offset += bytes;
}

static std::unique_ptr<std::vector<uint8_t>> listener_address_pack(
    const ListenerAddress& listener_address
) {
    size_t offset{0};
    size_t host_size = listener_address.host.size();
    const size_t total_size = sizeof(host_size) + host_size
                              + sizeof(listener_address.port)
                              + sizeof(listener_address.rank);
    auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

    auto encode = [&offset, &packed](const void* data, size_t bytes) {
        encode_(packed->data(), data, bytes, offset);
    };

    encode(&host_size, sizeof(host_size));
    encode(listener_address.host.data(), host_size);
    encode(&listener_address.port, sizeof(listener_address.port));
    encode(&listener_address.rank, sizeof(listener_address.rank));

    return packed;
}

static ListenerAddress listener_address_unpack(
    std::unique_ptr<std::vector<uint8_t>> packed
) {
    size_t offset{0};

    auto decode = [&offset, &packed](void* data, size_t bytes) {
        decode_(data, packed->data(), bytes, offset);
    };

    size_t host_size;
    decode(&host_size, sizeof(size_t));

    ListenerAddress listener_address;
    listener_address.host.resize(host_size);
    decode(listener_address.host.data(), host_size);

    decode(&listener_address.port, sizeof(listener_address.port));
    decode(&listener_address.rank, sizeof(listener_address.rank));

    return listener_address;
}

static std::unique_ptr<std::vector<uint8_t>> control_pack(
    rapidsmp::ControlMessage control, rapidsmp::ControlData data
) {
    size_t offset{0};
    const size_t total_size = sizeof(control) + get_size(data);

    // std::string packed(total_size, 0);
    auto packed = std::make_unique<std::vector<uint8_t>>(total_size);

    auto encode = [&offset, &packed](void const* data, size_t bytes) {
        encode_(packed->data(), data, bytes, offset);
    };

    encode(&control, sizeof(control));
    // std::visit(
    //     [&data, &encode](const ControlMessage& control) {
    //         using T = std::decay_t<decltype(control)>;
    //         if constexpr (std::is_same_v < T, ControlMessage::AssignRank>) {
    //             auto rank = std::get<Rank>(data);
    //             encode(&rank, sizeof(rank));
    //         } else if constexpr (std::is_same_v < T,
    //         ControlMessage::SetListenerAddress>)
    //         {
    //             auto listener_address = std::get<ListenerAddress>(data);
    //             encode(&listener_address, sizeof(listener_address));
    //         }
    //     },
    //     control
    // );
    if (control == ControlMessage::AssignRank
        || control == ControlMessage::GetListenerAddress)
    {
        auto rank = std::get<Rank>(data);
        encode(&rank, sizeof(rank));
    } else if (control == ControlMessage::SetListenerAddress) {
        auto listener_address = std::get<ListenerAddress>(data);
        auto packed_listener_address = listener_address_pack(listener_address);
        size_t packed_listener_address_size = packed_listener_address->size();
        encode(&packed_listener_address_size, sizeof(packed_listener_address_size));
        encode(packed_listener_address->data(), packed_listener_address_size);
    }

    std::cout << "Packed (" << packed->size() << " bytes): ";
    for (unsigned char i : *packed) {
        printf("\\0x%02x", i);
    }
    std::cout << std::endl;

    return packed;
};

static void control_unpack(
    std::shared_ptr<::ucxx::Buffer> buffer,
    ucp_ep_h ep,
    const ::ucxx::AmReceiverCallbackInfo& control_callback_info,
    std::shared_ptr<ListenerContainer> listener_container
) {
    size_t offset{0};

    auto decode = [&offset, &buffer](void* data, size_t bytes) {
        decode_(data, buffer->data(), bytes, offset);
    };

    ControlMessage control;
    decode(&control, sizeof(ControlMessage));

    // TODO: Fix logger
    // Logger& log = listener_container->logger();
    if (control == ControlMessage::AssignRank) {
        decode(listener_container->rank_.get(), sizeof(*listener_container->rank_));
        // log.warn("Received rank ", *listener_container->rank_);
        std::cout << "Received rank " << *listener_container->rank_ << std::endl;
    } else if (control == ControlMessage::RegisterEndpoint) {
        Rank rank;
        decode(&rank, sizeof(rank));
        (*listener_container->rank_to_endpoint_)[rank] =
            listener_container->endpoints_->at(ep);
    } else if (control == ControlMessage::SetListenerAddress) {
        size_t packed_listener_address_size;
        decode(&packed_listener_address_size, sizeof(packed_listener_address_size));
        auto packed_listener_address =
            std::make_unique<std::vector<uint8_t>>(packed_listener_address_size);
        decode(packed_listener_address->data(), packed_listener_address_size);
        ListenerAddress listener_address =
            listener_address_unpack(std::move(packed_listener_address));
        std::cout << "Rank " << listener_address.rank << " at address "
                  << listener_address.host << ":" << listener_address.port << std::endl;
        (*listener_container->rank_to_listener_address_)[listener_address.rank] =
            ListenerAddress{
                .host = listener_address.host,
                .port = listener_address.port,
                .rank = listener_address.rank,
            };

        // log.warn(
        //     "Rank ",
        //     listener_address.rank,
        //     " at address ",
        //     listener_address.host,
        //     ":",
        //     listener_address.port
        // );
    } else if (control == ControlMessage::GetListenerAddress) {
        Rank rank;
        decode(&rank, sizeof(rank));
        auto listener_address = listener_container->rank_to_listener_address_->at(rank);
        std::cout << "GetListenerAddress for rank " << rank << ", at address "
                  << listener_address.host << ":" << listener_address.port << " (rank "
                  << listener_address.rank << ")" << std::endl;
        auto endpoint = listener_container->endpoints_->at(ep);
        auto packed_listener_address =
            control_pack(ControlMessage::SetListenerAddress, listener_address);
        auto req = endpoint->amSend(
            packed_listener_address->data(),
            packed_listener_address->size(),
            UCS_MEMORY_TYPE_HOST,
            control_callback_info
        );
        // TODO: Wait for completion
        assert(req->isCompleted());
    }
};

static void listener_callback(ucp_conn_request_h conn_request, void* arg) {
    auto listener_container = reinterpret_cast<ListenerContainer*>(arg);

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
    std::cout << "Server received a connection request from client at address "
              << ip_str.data() << ":" << port_str.data() << std::endl;

    auto endpoint =
        listener_container->listener_->createEndpointFromConnRequest(conn_request, true);
    (*listener_container->endpoints_)[endpoint->getHandle()] = endpoint;


    if (listener_container->root_) {
        // TODO: Reuse receive_rank_callback_info
        // ::ucxx::AmReceiverCallbackInfo receive_rank_callback_info("rapidsmp", 0);
        // TODO: Ensure nextRank remains alive until request completes
        Rank client_rank = listener_container->get_next_worker_rank_();
        auto packed_client_rank = control_pack(ControlMessage::AssignRank, client_rank);
        std::cout << "Assigning rank " << client_rank << " to client at address "
                  << ip_str.data() << ":" << port_str.data() << std::endl;
        std::cout << "Packed (returned, " << packed_client_rank->size() << " bytes): ";
        for (unsigned char i : *packed_client_rank) {
            printf("\\0x%02x", i);
        }
        std::cout << std::endl;
        auto req = endpoint->amSend(
            packed_client_rank->data(),
            packed_client_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            *listener_container->control_callback_info_
        );
        listener_container->futures_->push_back(
            std::make_unique<HostFuture>(req, std::move(packed_client_rank))
        );
        (*listener_container->rank_to_endpoint_)[client_rank] = endpoint;
    }
}

UCXX::UCXX(std::shared_ptr<::ucxx::Worker> worker, std::uint32_t nranks)
    : worker_(std::move(worker)),
      listener_container_(std::make_shared<ListenerContainer>()),
      endpoints_(std::make_shared<EndpointsMap>()),
      rank_to_endpoint_(std::make_shared<RankToEndpointMap>()),
      rank_to_listener_address_(std::make_shared<RankToListenerAddressMap>()),
      rank_(std::make_shared<Rank>(0)),
      control_callback_info_(
          std::make_shared<::ucxx::AmReceiverCallbackInfo>("rapidsmp", 0)
      ),
      nranks_(nranks),
      next_rank_(Rank(0)),
      logger_(this) {
    if (worker_ == nullptr) {
        auto context = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
        worker_ = context->createWorker(false);
        // TODO: Allow other modes
        worker_->startProgressThread(true);
    }

    // Create listener
    listener_ = worker_->createListener(0, listener_callback, listener_container_.get());
    listener_container_->listener_ = listener_;
    listener_container_->endpoints_ = endpoints_;
    listener_container_->rank_ = rank_;
    listener_container_->rank_to_endpoint_ = rank_to_endpoint_;
    listener_container_->rank_to_listener_address_ = rank_to_listener_address_;
    listener_container_->root_ = true;
    listener_container_->control_callback_info_ = control_callback_info_;
    listener_container_->futures_ =
        std::make_shared<std::vector<std::unique_ptr<HostFuture>>>();
    listener_container_->get_next_worker_rank_ = [this]() {
        return get_next_worker_rank();
    };
    // listener_container_->logger = [this]() {
    //     return logger();
    // };
    // listener_container_->log = logger();

    auto listener_address = ListenerAddress{
        .host = listener_->getIp(), .port = listener_->getPort(), .rank = *rank_
        // .host = "localhost", .port = listener_->getPort(), .rank = *rank_
    };
    (*rank_to_listener_address_)[*rank_] = listener_address;

    Logger& log = logger();
    log.warn(
        "Root running at address ", listener_address.host, ":", listener_address.port
    );

    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            control_unpack(
                req->getRecvBuffer(), ep, *control_callback_info_, listener_container_
            );
        }
    );

    worker_->registerAmReceiverCallback(*control_callback_info_, control_callback);
}

UCXX::UCXX(
    std::shared_ptr<::ucxx::Worker> worker,
    std::uint32_t nranks,
    std::string root_host,
    uint16_t root_port
)
    : worker_(std::move(worker)),
      listener_container_(std::make_shared<ListenerContainer>()),
      endpoints_(std::make_shared<EndpointsMap>()),
      rank_to_endpoint_(std::make_shared<RankToEndpointMap>()),
      rank_to_listener_address_(std::make_shared<RankToListenerAddressMap>()),
      rank_(std::make_shared<Rank>(-1)),
      control_callback_info_(
          std::make_shared<::ucxx::AmReceiverCallbackInfo>("rapidsmp", 0)
      ),
      nranks_(nranks),
      next_rank_(Rank(-1)),
      logger_(this) {
    if (worker_ == nullptr) {
        auto context = ::ucxx::createContext({}, ::ucxx::Context::defaultFeatureFlags);
        worker_ = context->createWorker(false);
        // TODO: Allow other modes
        worker_->startProgressThread(true);
    }

    // Create listener
    listener_ = worker_->createListener(0, listener_callback, listener_container_.get());
    listener_container_->listener_ = listener_;
    listener_container_->endpoints_ = endpoints_;
    listener_container_->rank_ = rank_;
    listener_container_->rank_to_endpoint_ = rank_to_endpoint_;
    listener_container_->rank_to_listener_address_ = rank_to_listener_address_;
    listener_container_->root_ = false;
    listener_container_->control_callback_info_ = control_callback_info_;
    listener_container_->get_next_worker_rank_ = [this]() {
        return get_next_worker_rank();
    };
    listener_container_->futures_ =
        std::make_shared<std::vector<std::unique_ptr<HostFuture>>>();

    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            auto buffer = req->getRecvBuffer();
            auto data = static_cast<const unsigned char*>(buffer->data());
            std::cout << "Received buffer (callback, " << buffer->getSize()
                      << " bytes): ";
            for (size_t i = 0; i < buffer->getSize(); ++i) {
                // printf("\\0x%02x", data[i]);
                printf("\\0x%02x", data[i]);
            }
            std::cout << std::endl;
            control_unpack(
                req->getRecvBuffer(), ep, *control_callback_info_, listener_container_
            );
        }
    );

    worker_->registerAmReceiverCallback(*control_callback_info_, control_callback);

    // Connect to root
    Logger& log = logger();
    log.warn(
        "Connecting to root node at ", root_host, ":", root_port, "current rank: ", *rank_
    );
    auto endpoint = worker_->createEndpointFromHostname(root_host, root_port, true);
    (*rank_to_endpoint_)[Rank(0)] = endpoint;
    (*endpoints_)[endpoint->getHandle()] = endpoint;

    // Get my rank
    while (*rank_ == Rank(-1)) {
        // TODO: progress in non-progress thread modes
    }
    log.warn("My new rank: ", *rank_);

    // Inform listener address
    ListenerAddress listener_address = ListenerAddress{
        .host = "localhost", .port = listener_->getPort(), .rank = *rank_
    };
    auto packed_listener_address =
        control_pack(ControlMessage::SetListenerAddress, listener_address);
    auto req = endpoint->amSend(
        packed_listener_address->data(),
        packed_listener_address->size(),
        UCS_MEMORY_TYPE_HOST,
        // receive_listener_address_callback_info
        *control_callback_info_
    );
    // TODO: Wait for completion
    assert(req->isCompleted());
}

static ::ucxx::Tag tag_with_rank(Rank rank, int tag) {
    // The rapidsmp::Communicator API uses 32-bit `int` for user tags to match
    // MPI's standard. We can thus pack the rank in the higher 32-bit of UCX's
    // 64-bit tags as aid in identifying the sender of a message. Since we're
    // currently limited to 26-bits for ranks (see
    // `rapidsmp::shuffler::Shuffler::get_new_cid()`), we are essentially using
    // 58-bits for the tags and the remaining 6-bits may be used in the future,
    // such as to identify groups.
    return ::ucxx::Tag(static_cast<uint64_t>(rank) << 32 | tag);
}

static constexpr ::ucxx::TagMask UserTagMask{std::numeric_limits<int>::max()};

std::shared_ptr<::ucxx::Endpoint> UCXX::get_endpoint(Rank rank) {
    Logger& log = logger();
    try {
        log.warn("Endpoint for rank ", rank, " already available, returning to caller");
        return rank_to_endpoint_->at(rank);
    } catch (std::out_of_range const&) {
        log.warn(
            "Endpoint for rank ", rank, " not available, requesting listener address"
        );
        auto packed_rank = control_pack(ControlMessage::GetListenerAddress, rank);


        std::cout << "Packed (returned, " << packed_rank->size() << " bytes): ";
        for (unsigned char i : *packed_rank) {
            printf("\\0x%02x", i);
        }
        std::cout << std::endl;

        auto root_endpoint = get_endpoint(Rank(0));

        auto req = root_endpoint->amSend(
            packed_rank->data(),
            packed_rank->size(),
            UCS_MEMORY_TYPE_HOST,
            *control_callback_info_
        );

        // while (!req->isCompleted() && rank_to_listener_address_->find(rank) ==
        // rank_to_listener_address_->end()) ;
        while (!req->isCompleted())
            ;
        while (rank_to_listener_address_->find(rank) == rank_to_listener_address_->end())
            ;

        auto& listener_address = rank_to_listener_address_->at(rank);
        auto endpoint = worker_->createEndpointFromHostname(
            listener_address.host, listener_address.port, true
        );
        (*rank_to_endpoint_)[rank] = endpoint;
        (*endpoints_)[endpoint->getHandle()] = endpoint;

        log.warn(
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
    auto req =
        get_endpoint(rank)->tagSend(msg->data(), msg->size(), tag_with_rank(*rank_, tag));
    return std::make_unique<Future>(req, br->move(std::move(msg), stream));
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream
) {
    auto req =
        get_endpoint(rank)->tagSend(msg->data(), msg->size, tag_with_rank(*rank_, tag));
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> UCXX::recv(
    Rank rank, int tag, std::unique_ptr<Buffer> recv_buffer, rmm::cuda_stream_view stream
) {
    auto req = get_endpoint(rank)->tagRecv(
        recv_buffer->data(), recv_buffer->size, ::ucxx::Tag(tag), UserTagMask
    );
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> UCXX::recv_any(int tag) {
    Logger& log = logger();
    // int msg_available;
    // MPI_Status probe_status;
    // RAPIDSMP_MPI(MPI_Iprobe(MPI_ANY_SOURCE, tag, comm_, &msg_available,
    // &probe_status));
    auto probe = worker_->tagProbe(::ucxx::Tag(tag), UserTagMask);
    auto msg_available = probe.first;
    auto info = probe.second;
    auto sender_rank = static_cast<Rank>(info.senderTag >> 32);
    if (!msg_available) {
        return {nullptr, 0};
    }
    // RAPIDSMP_EXPECTS(
    //     tag == probe_status.MPI_TAG || tag == MPI_ANY_TAG, "corrupt mpi tag"
    // );
    // MPI_Count size;
    // RAPIDSMP_MPI(MPI_Get_elements_x(&probe_status, MPI_UINT8_T, &size));
    auto msg = std::make_unique<std::vector<uint8_t>>(info.length);  // TODO: uninitialize

    // MPI_Status msg_status;
    // RAPIDSMP_MPI(MPI_Recv(
    //     msg->data(),
    //     msg->size(),
    //     MPI_UINT8_T,
    //     probe_status.MPI_SOURCE,
    //     probe_status.MPI_TAG,
    //     comm_,
    //     &msg_status
    // ));
    // RAPIDSMP_MPI(MPI_Get_elements_x(&msg_status, MPI_UINT8_T, &size));
    // RAPIDSMP_EXPECTS(
    //     static_cast<std::size_t>(size) == msg->size(),
    //     "incorrect size of the MPI_Recv message"
    // );
    auto req = worker_->tagRecv(msg->data(), msg->size(), ::ucxx::Tag(tag), UserTagMask);
    // if (msg->size() > 2048) {  // TODO: use the actual eager threshold.
    //     log.warn(
    //         "block-receiving a messager larger than the normal ",
    //         "eager threshold (",
    //         msg->size(),
    //         " bytes)"
    //     );
    // }
    if (!req->isCompleted()) {
        log.warn(
            "block-receiving a messager larger than the normal ",
            "eager threshold (",
            msg->size(),
            " bytes)"
        );
        // TODO: PROGRESS
    }
    // return {std::move(msg), probe_status.MPI_SOURCE};
    // TODO: Fix sender rank
    return {std::move(msg), sender_rank};
}

std::vector<std::size_t> UCXX::test_some(
    std::vector<std::unique_ptr<Communicator::Future>> const& future_vector
) {
    // TODO: Progress before checking completion

    std::vector<std::shared_ptr<::ucxx::Request>> reqs;
    for (auto const& future : future_vector) {
        auto ucxx_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMP_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
        reqs.push_back(ucxx_future->req_);
    }

    // Get completed requests as indices into `future_vector` (and `reqs`).
    std::vector<size_t> completed;
    for (size_t i = 0; i < reqs.size(); ++i) {
        if (reqs[i]->isCompleted())
            completed.push_back(i);
    }
    return completed;
}

std::vector<std::size_t> UCXX::test_some(
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
        future_map
) {
    std::vector<std::shared_ptr<::ucxx::Request>> reqs;
    std::vector<std::size_t> key_reqs;
    reqs.reserve(future_map.size());
    key_reqs.reserve(future_map.size());
    for (auto const& [key, future] : future_map) {
        auto ucxx_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMP_EXPECTS(ucxx_future != nullptr, "future isn't a UCXX::Future");
        reqs.push_back(ucxx_future->req_);
        key_reqs.push_back(key);
    }

    // Get completed requests as indices into `key_reqs` (and `reqs`).
    std::vector<size_t> completed;
    for (size_t i = 0; i < reqs.size(); ++i) {
        if (reqs[i]->isCompleted())
            completed.push_back(i);
    }

    std::vector<std::size_t> ret;
    ret.reserve(completed.size());
    for (size_t i : completed) {
        ret.push_back(key_reqs.at(i));
    }
    return ret;
}

void UCXX::barrier() {
    Logger& log = logger();
    log.warn("Barrier started on rank ", *rank_);
    while (*rank_ == 0
           && rank_to_listener_address_->size() != static_cast<uint64_t>(nranks()))
    {
        // TODO: Update progress mode
    }

    if (*rank_ == 0) {
        std::vector<std::shared_ptr<::ucxx::Request>> requests;
        for (auto& rank_to_endpoint : *rank_to_endpoint_) {
            auto& endpoint = rank_to_endpoint.second;
            requests.push_back(endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST));
        }
        // TODO: Update progress mode
        while (std::all_of(requests.begin(), requests.end(), [](const auto& req) {
            return !req->isCompleted();
        }))
            ;
        requests.clear();
        for (auto& rank_to_endpoint : *rank_to_endpoint_) {
            auto& endpoint = rank_to_endpoint.second;
            requests.push_back(endpoint->amRecv());
        }
        // TODO: Update progress mode
        while (std::all_of(requests.begin(), requests.end(), [](const auto& req) {
            return !req->isCompleted();
        }))
            ;
    } else {
        auto endpoint = get_endpoint(0);
        auto req = endpoint->amRecv();
        // TODO: Update progress mode
        while (!req->isCompleted())
            ;
        req = endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST);
        // TODO: Update progress mode
        while (!req->isCompleted())
            ;
    }
    log.warn("Barrier completed on rank ", *rank_);
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
    ss << "UCXX(rank=" << *rank_ << ", nranks: " << nranks_ << ", ucx-version=" << major
       << "." << minor << "." << release << ")";
    return ss.str();
}

Rank UCXX::get_next_worker_rank() {
    if (*rank_ != 0)
        throw std::runtime_error("This method can only be called by rank 0");
    return ++next_rank_;
}

UCXX::~UCXX() noexcept {
    Logger& log = logger();
    log.warn("~UCXX");
    worker_->stopProgressThread();
}

}  // namespace rapidsmp
