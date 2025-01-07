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

static std::string control_pack(
    rapidsmp::ControlMessage control, rapidsmp::ControlData data
) {
    size_t offset{0};
    const size_t total_size = sizeof(control) + get_size(data);

    std::string packed(total_size, 0);

    auto encode = [&offset, &packed](void const* data, size_t bytes) {
        std::cout << "Encoding " << bytes << " bytes" << std::endl;
        for (size_t i = 0; i < bytes; ++i) {
            printf("\\0x%02x", static_cast<const unsigned char*>(data)[i]);
        }
        std::cout << std::endl;
        memcpy(packed.data() + offset, data, bytes);
        offset += bytes;
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
    if (control == ControlMessage::AssignRank) {
        auto rank = std::get<Rank>(data);
        encode(&rank, sizeof(rank));
    } else if (control == ControlMessage::SetListenerAddress) {
        auto listener_address = std::get<ListenerAddress>(data);
        encode(&listener_address, sizeof(listener_address));
    }

    std::cout << "Packed (" << packed.size() << " bytes): " << packed;
    for (char i : packed) {
        printf("\\0x%02x", i);
    }
    std::cout << std::endl;

    return packed;
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
        ::ucxx::AmReceiverCallbackInfo control_callback_info("rapidsmp", 0);
        // TODO: Ensure nextRank remains alive until request completes
        Rank client_rank = listener_container->get_next_worker_rank_();
        auto packed_client_rank = control_pack(ControlMessage::AssignRank, client_rank);
        std::cout << "Assigning rank " << client_rank << " to client at address "
                  << ip_str.data() << ":" << port_str.data() << std::endl;
        std::cout << "Packed (returned, " << packed_client_rank.size() << " bytes): ";
        for (char i : packed_client_rank) {
            printf("\\0x%02x", i);
        }
        std::cout << std::endl;
        auto req = endpoint->amSend(
            packed_client_rank.data(),
            packed_client_rank.size(),
            UCS_MEMORY_TYPE_HOST,
            control_callback_info
        );
        (*listener_container->rank_to_endpoint_)[client_rank] = endpoint;
    }
}

UCXX::UCXX(std::shared_ptr<::ucxx::Worker> worker, std::uint32_t nranks)
    : worker_(std::move(worker)),
      endpoints_(std::make_shared<EndpointsMap>()),
      rank_to_endpoint_(std::make_shared<RankToEndpointMap>()),
      rank_to_listener_address_(std::make_shared<RankToListenerAddressMap>()),
      rank_(Rank(0)),
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
    listener_ = worker_->createListener(0, listener_callback, &listener_container_);
    listener_container_.listener_ = listener_;
    listener_container_.endpoints_ = endpoints_;
    listener_container_.rank_to_endpoint_ = rank_to_endpoint_;
    listener_container_.root_ = true;
    listener_container_.get_next_worker_rank_ = [this]() {
        return get_next_worker_rank();
    };

    Logger& log = logger();
    log.warn(
        "Server received a connection request from client at address ",
        listener_->getIp(),
        ":",
        listener_->getPort()
    );

    ::ucxx::AmReceiverCallbackInfo control_callback_info("rapidsmp", 0);

    auto control_unpack = [this](
                              std::shared_ptr<::ucxx::Buffer> buffer,
                              ucp_ep_h ep,
                              const ::ucxx::AmReceiverCallbackInfo& control_callback_info
                          ) {
        size_t offset{0};

        auto decode = [&offset, &buffer](void* data, size_t bytes) {
            memcpy(data, static_cast<const char*>(buffer->data()) + offset, bytes);
            offset += bytes;
        };

        ControlMessage control;
        decode(&control, sizeof(ControlMessage));

        Logger& log = logger();
        if (control == ControlMessage::AssignRank) {
            decode(&rank_, sizeof(rank_));
            log.warn("Received rank ", rank_);
        } else if (control == ControlMessage::RegisterEndpoint) {
            Rank rank;
            decode(&rank, sizeof(rank));
            (*rank_to_endpoint_)[rank] = endpoints_->at(ep);
        } else if (control == ControlMessage::SetListenerAddress) {
            ListenerAddress listener_address;
            decode(&listener_address, sizeof(listener_address));
            (*rank_to_listener_address_)[listener_address.rank] = listener_address;
            log.warn(
                "Rank ",
                listener_address.rank,
                " at address ",
                listener_address.host,
                ":",
                listener_address.port
            );
        } else if (control == ControlMessage::GetListenerAddress) {
            Rank rank;
            decode(&rank, sizeof(rank));
            auto listener_address = rank_to_listener_address_->at(rank);
            auto endpoint = endpoints_->at(ep);
            auto packed_listener_address =
                control_pack(ControlMessage::SetListenerAddress, listener_address);
            auto req = endpoint->amSend(
                packed_listener_address.data(),
                packed_listener_address.size(),
                UCS_MEMORY_TYPE_HOST,
                control_callback_info
            );
            // TODO: Wait for completion
            assert(req->isCompleted());
        }
    };

    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [&control_unpack, &control_callback_info](
            std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep
        ) { control_unpack(req->getRecvBuffer(), ep, control_callback_info); }
    );

    worker_->registerAmReceiverCallback(control_callback_info, control_callback);
}

UCXX::UCXX(
    std::shared_ptr<::ucxx::Worker> worker,
    std::uint32_t nranks,
    std::string root_host,
    uint16_t root_port
)
    : worker_(std::move(worker)),
      endpoints_(std::make_shared<EndpointsMap>()),
      rank_to_endpoint_(std::make_shared<RankToEndpointMap>()),
      rank_to_listener_address_(std::make_shared<RankToListenerAddressMap>()),
      rank_(Rank(-1)),
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
    listener_ = worker_->createListener(0, listener_callback, &listener_container_);
    listener_container_.listener_ = listener_;
    listener_container_.endpoints_ = endpoints_;
    listener_container_.rank_to_endpoint_ = rank_to_endpoint_;
    listener_container_.root_ = false;
    listener_container_.get_next_worker_rank_ = [this]() {
        return get_next_worker_rank();
    };

    ::ucxx::AmReceiverCallbackInfo control_callback_info("rapidsmp", 0);

    auto control_unpack = [this](
                              std::shared_ptr<::ucxx::Buffer> buffer,
                              ucp_ep_h ep,
                              const ::ucxx::AmReceiverCallbackInfo& control_callback_info
                          ) {
        size_t offset{0};

        auto decode = [&offset, &buffer](void* data, size_t bytes) {
            memcpy(data, static_cast<const char*>(buffer->data()) + offset, bytes);
            offset += bytes;
        };

        Logger& log = logger();

        std::cout << "Received buffer (" << buffer->getSize() << " bytes): ";
        for (size_t i = 0; i < buffer->getSize(); ++i) {
            printf("\\0x%02x", static_cast<const unsigned char*>(buffer->data())[i]);
        }
        std::cout << std::endl;

        ControlMessage control;
        log.warn("Message type before receiving: ", static_cast<size_t>(control));
        decode(&control, sizeof(ControlMessage));

        log.warn("Received control message of type: ", static_cast<size_t>(control));

        if (control == ControlMessage::AssignRank) {
            decode(&rank_, sizeof(rank_));
            log.warn("Received rank ", rank_);
        } else if (control == ControlMessage::RegisterEndpoint) {
            Rank rank;
            decode(&rank, sizeof(rank));
            (*rank_to_endpoint_)[rank] = endpoints_->at(ep);
        } else if (control == ControlMessage::SetListenerAddress) {
            ListenerAddress listener_address;
            decode(&listener_address, sizeof(listener_address));
            (*rank_to_listener_address_)[listener_address.rank] = listener_address;
            log.warn(
                "Rank ",
                listener_address.rank,
                " at address ",
                listener_address.host,
                ":",
                listener_address.port
            );
        } else if (control == ControlMessage::GetListenerAddress) {
            Rank rank;
            decode(&rank, sizeof(rank));
            auto listener_address = rank_to_listener_address_->at(rank);
            auto endpoint = endpoints_->at(ep);
            auto packed_listener_address =
                control_pack(ControlMessage::SetListenerAddress, listener_address);
            auto req = endpoint->amSend(
                packed_listener_address.data(),
                packed_listener_address.size(),
                UCS_MEMORY_TYPE_HOST,
                control_callback_info
            );
            // TODO: Wait for completion
            assert(req->isCompleted());
        }
    };

    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [&control_unpack,
         &control_callback_info](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            auto buffer = req->getRecvBuffer();
            auto data = static_cast<const unsigned char*>(buffer->data());
            std::cout << "Received buffer (callback, " << buffer->getSize()
                      << " bytes): ";
            for (size_t i = 0; i < buffer->getSize(); ++i) {
                // printf("\\0x%02x", data[i]);
                printf("\\0x%02x", data[i]);
            }
            std::cout << std::endl;
            control_unpack(req->getRecvBuffer(), ep, control_callback_info);
        }
    );

    worker_->registerAmReceiverCallback(control_callback_info, control_callback);

    // Connect to root
    Logger& log = logger();
    log.warn(
        "Connecting to root node at ", root_host, ":", root_port, "current rank: ", rank_
    );
    auto endpoint = worker_->createEndpointFromHostname(root_host, root_port, true);
    (*rank_to_endpoint_)[Rank(0)] = endpoint;
    (*endpoints_)[endpoint->getHandle()] = endpoint;

    // Get my rank
    while (rank_ == Rank(-1)) {
        // TODO: progress in non-progress thread modes
    }
    log.warn("My new rank: ", rank_);

    // Inform listener address
    ListenerAddress listener_address =
        ListenerAddress{.host = "localhost", .port = listener_->getPort(), .rank = rank_};
    auto packed_listener_address =
        control_pack(ControlMessage::SetListenerAddress, listener_address);
    auto req = endpoint->amSend(
        packed_listener_address.data(),
        packed_listener_address.size(),
        UCS_MEMORY_TYPE_HOST,
        // receive_listener_address_callback_info
        control_callback_info
    );
    // TODO: Wait for completion
    assert(req->isCompleted());
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<std::vector<uint8_t>> msg,
    Rank rank,
    int tag,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    auto req =
        rank_to_endpoint_->at(rank)->tagSend(msg->data(), msg->size(), ::ucxx::Tag(tag));
    return std::make_unique<Future>(req, br->move(std::move(msg), stream));
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream
) {
    auto req =
        rank_to_endpoint_->at(rank)->tagSend(msg->data(), msg->size, ::ucxx::Tag(tag));
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> UCXX::recv(
    Rank rank, int tag, std::unique_ptr<Buffer> recv_buffer, rmm::cuda_stream_view stream
) {
    auto req = rank_to_endpoint_->at(rank)->tagRecv(
        recv_buffer->data(), recv_buffer->size, ::ucxx::Tag(tag), ::ucxx::TagMaskFull
    );
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> UCXX::recv_any(int tag) {
    Logger& log = logger();
    // int msg_available;
    // MPI_Status probe_status;
    // RAPIDSMP_MPI(MPI_Iprobe(MPI_ANY_SOURCE, tag, comm_, &msg_available,
    // &probe_status));
    auto probe = worker_->tagProbe(::ucxx::Tag(tag));
    auto msg_available = probe.first;
    auto info = probe.second;
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
    auto req =
        worker_->tagRecv(msg->data(), msg->size(), ::ucxx::Tag(tag), ::ucxx::TagMaskFull);
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
    return {std::move(msg), 0};
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
    while (rank_ == 0 && rank_to_endpoint_->size() != static_cast<uint64_t>(nranks())) {
        // TODO: Update progress mode
    }

    if (rank_ == 0) {
        std::vector<std::shared_ptr<::ucxx::Request>> requests;
        for (auto& rank_to_endpoint : *rank_to_endpoint_) {
            auto& endpoint = rank_to_endpoint.second;
            requests.push_back(endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST));
        }
        // TODO: Update progress mode
        while (std::all_of(requests.begin(), requests.end(), [](const auto& req) {
            return req->isCompleted();
        }))
            ;
        requests.clear();
        for (auto& rank_to_endpoint : *rank_to_endpoint_) {
            auto& endpoint = rank_to_endpoint.second;
            requests.push_back(endpoint->amRecv());
        }
        // TODO: Update progress mode
        while (std::all_of(requests.begin(), requests.end(), [](const auto& req) {
            return req->isCompleted();
        }))
            ;
    } else {
        auto& endpoint = rank_to_endpoint_->at(0);
        auto req = endpoint->amRecv();
        // TODO: Update progress mode
        while (!req->isCompleted())
            ;
        req = endpoint->amSend(nullptr, 0, UCS_MEMORY_TYPE_HOST);
        // TODO: Update progress mode
        while (!req->isCompleted())
            ;
    }
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
    ss << "UCXX(rank=" << rank_ << ", nranks: " << nranks_ << ", ucx-version=" << major
       << "." << minor << "." << release << ")";
    return ss.str();
}

Rank UCXX::get_next_worker_rank() {
    if (rank_ != 0)
        throw std::runtime_error("This method can only be called by rank 0");
    return ++next_rank_;
}

}  // namespace rapidsmp
