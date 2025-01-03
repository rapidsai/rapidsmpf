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

#include <rapidsmp/communicator/mpi.hpp>
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

static void listener_callback(ucp_conn_request_h conn_request, void* arg) {
    auto listener_container = reinterpret_cast<ListenerContainer*>(arg);

    ucp_conn_request_attr_t attr{};
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    auto status = ucp_conn_request_query(
        conn_request, &attr
    );  // TODO: Log if creating endpoint failed?
    if (status != UCS_OK)
        return;

    auto endpoint =
        listener_container->listener_->createEndpointFromConnRequest(conn_request, true);
    listener_container->endpoints_->at(endpoint->getHandle()) = endpoint;

    if (listener_container->root_) {
        // TODO: Reuse receive_rank_callback_info
        // ::ucxx::AmReceiverCallbackInfo receive_rank_callback_info("rapidsmp", 0);
        ::ucxx::AmReceiverCallbackInfo control_callback_info("rapidsmp", 0);
        // TODO: Ensure nextRank remains alive until request completes
        Rank client_rank = get_next_worker_rank();
        auto req = endpoint->amSend(
            &client_rank, sizeof(client_rank), UCS_MEMORY_TYPE_HOST, control_callback_info
        );
        listener_container->rank_to_endpoint_->at(client_rank) = endpoint;
    }
}

UCXX::UCXX(std::shared_ptr<::ucxx::Worker> worker, bool root, std::uint32_t nranks)
    : worker_(worker),
      endpoints_(std::make_shared<EndpointsMap>()),
      rank_to_endpoint_(std::make_shared<RankToEndpointMap>()),
      rank_to_listener_address_(std::make_shared<RankToListenerAddressMap>()),
      rank_(Rank(-1)),
      nranks_(nranks),
      next_rank_(Rank(0)),
      logger_(this) {
    // int rank;
    // int nranks;
    // RAPIDSMP_MPI(MPI_Comm_rank(comm_, &rank));
    // RAPIDSMP_MPI(MPI_Comm_size(comm_, &nranks));
    // rank_ = rank;
    // nranks_ = nranks;
    // check_mpi_thread_support();

    if (worker_ == nullptr) {
        auto context = ::ucxx::createContext({{}}, ::ucxx::Context::defaultFeatureFlags);
        worker_ = context->createWorker(false);
        // TODO: Allow other modes
        worker_ = context->startProgressThread(true);
    }

    // Create listener
    listener_ = worker_->createListener(0, listener_callback, &listener_container_);
    listener_container_.listener_ = listener_;
    listener_container_.endpoints_ = endpoints_;
    listener_container_.rank_to_endpoint_ = rank_to_endpoint_;
    listener_container_.root_ = root;

    // ::ucxx::AmReceiverCallbackInfo receive_rank_callback_info("rapidsmp", 0);
    // auto receive_rank = ::ucxx::AmReceiverCallbackType(
    //     [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
    //         rank_ = *reinterpret_cast<Rank*>(req->getRecvBuffer()->data());
    //     }
    // );

    // ::ucxx::AmReceiverCallbackInfo endpoint_registration_callback_info("rapidsmp", 1);
    // auto endpoint_registration_callback = ::ucxx::AmReceiverCallbackType(
    //     [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
    //         auto rank = *reinterpret_cast<Rank*>(req->getRecvBuffer()->data());
    //         rank_to_endpoint_[rank] = ep;
    //     }
    // );

    // ::ucxx::AmReceiverCallbackInfo receive_listener_address_callback_info(
    //     "rapidsmp", 2
    // );
    // auto receive_listener_address =
    //     ::ucxx::AmReceiverCallbackType(
    //         [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
    //             auto listener_address =
    //                 reinterpret_cast<ListenerAddress*>(req->getRecvBuffer()->data());
    //             rank_to_listener_address_[listener_address->rank] = ListenerAddress {
    //                 .host = listener_address->host;
    //                 .port = listener_address->host;
    //                 .rank = listener_address->rank;
    //             }
    //         }
    //     );

    // worker_->registerAmReceiverCallback(
    //     endpoint_registration_callback_info, endpoint_registration_callback
    // );
    // worker_->registerAmReceiverCallback(receive_rank_callback_info, receive_rank);
    // worker_->registerAmReceiverCallback(
    //     receive_listener_address_callback_info, receive_listener_address
    // );

    auto get_size = [](const auto& data) {
        return std::visit(
            [](const auto& data) { return sizeof(std::decay_t<decltype(data)>); }, data
        );
    };

    auto control_pack =
        [](ControlMessage control, ControlData data) {
            size_t offset{0};
            const size_t total_size = sizeof(control) + get_size(data);

            std::string packed(total_size, 0);

            auto encode =
                [&offset, &packed](void const* data, size_t bytes) {
                    memcpy(packed.data() + offset, data, bytes);
                    offset += bytes;
                }

            encode(&control, sizeof(control));
            std::visit(
                [&data](const ControlMessage& control) {
                    using T = std::decay_t<decltype(control)>;
                    if constexpr (std::is_same_v < T, ControlMessage::AssignRank) {
                        auto rank = std::get<Rank>(data);
                        encode(&rank, sizeof(rank));
                    } else if constexpr (std::is_same_v < T, ControlMessage::SetListenerAddress)
                    {
                        auto listener_address = std::get<ListenerAddress>(data);
                        encode(&listener_address, sizeof(listener_address));
                    }
                },
                control
            );
        }

    auto control_unpack =
        [this](std::shared_ptr<::ucxx::HostBuffer> buffer, ucp_ep_h ep) {
            size_t offset{0};

            auto decode =
                [&offset, &buffer](void* data, size_t bytes) {
                    memcpy(data, buffer->data() + offset, bytes);
                    offset += bytes;
                }

            ControlMessage control;
            decode(&control, sizeof(ControlMessage));

            std::visit(
                [this](const ControlMessage& control) {
                    Logger& log = logger();

                    using T = std::decay_t<decltype(control)>;
                    if constexpr (std::is_same_v < T, ControlMessage::AssignRank) {
                        decode(&rank_, sizeof(rank_));
                        log.warn("Received rank ", rank_);
                    } else if constexpr (std::is_same_v < T, ControlMessage::RegisterEndpoint)
                    {
                        Rank rank;
                        decode(&rank, sizeof(rank));
                        rank_to_endpoint_[rank] = ep;
                    } else if constexpr (std::is_same_v < T, ControlMessage::SetListenerAddress)
                    {
                        ListenerAddress listener_address;
                        decode(&listener_address, sizeof(listener_address));
                        rank_to_listener_address_[listener_address->rank] =
                            listener_address;
                        log.warn(
                            "Rank ",
                            listener_address->rank,
                            " at address ",
                            listener_address->host,
                            ":",
                            listener_address->port
                        );
                    } else if constexpr (std::is_save_v < T, ControlMessage::GetListenerAddress)
                    {
                        Rank rank;
                        decode(&rank, sizeof(rank));
                        auto listener_address = rank_to_listener_address_[rank];
                        endpoint->amSend(
                            static_cast<void*>(&listener_address),
                            sizeof(listener_address),
                            UCS_MEMORY_TYPE_HOST,
                            control_callback_info
                        );
                    }
                },
                control
            );
        }

    ::ucxx::AmReceiverCallbackInfo control_callback_info("rapidsmp", 0);
    auto control_callback = ::ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<::ucxx::Request> req, ucp_ep_h ep) {
            control_unpack(req->getRecvBuffer(), ep);
        }
    );

    worker_->registerAmReceiverCallback(control_callback_info, control_callback);

    if (root) {
        rank_ = Rank(0);
    } else {
        // Connect to root
        auto endpoint = worker_->createEndpointFromHostname("localhost", port, true);
        rank_to_endpoint_[Rank(0)] = endpoint->getHandle();
        endpoints_[endpoint->getHandle()] = endpoint;

        // Get my rank
        while (rank_ == Rank(-1)) {
            // TODO: progress in non-progress thread modes
        }

        // Inform listener address
        ListenerAddress listener_address = ListenerAddress{
            .host = "localhost", .port = listener_->getPort(), .rank = rank_
        };
        endpoint->amSend(
            static_cast<void*>(&listener_address),
            sizeof(listener_address),
            UCS_MEMORY_TYPE_HOST,
            // receive_listener_address_callback_info
            control_callback_info
        );
    }
}

std::unique_ptr<Communicator::Future> UCXX::send(
    std::unique_ptr<std::vector<uint8_t>> msg,
    Rank rank,
    int tag,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
    auto req = MPI_Request req;
    RAPIDSMP_MPI(MPI_Isend(msg->data(), msg->size(), MPI_UINT8_T, rank, tag, comm_, &req)
    );
    return std::make_unique<Future>(req, br->move(std::move(msg), stream));
}

std::unique_ptr<Communicator::Future> MPI::send(
    std::unique_ptr<Buffer> msg, Rank rank, int tag, rmm::cuda_stream_view stream
) {
    MPI_Request req;
    RAPIDSMP_MPI(MPI_Isend(msg->data(), msg->size, MPI_UINT8_T, rank, tag, comm_, &req));
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> MPI::recv(
    Rank rank, int tag, std::unique_ptr<Buffer> recv_buffer, rmm::cuda_stream_view stream
) {
    MPI_Request req;
    RAPIDSMP_MPI(MPI_Irecv(
        recv_buffer->data(), recv_buffer->size, MPI_UINT8_T, rank, tag, comm_, &req
    ));
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> MPI::recv_any(int tag) {
    Logger& log = logger();
    int msg_available;
    MPI_Status probe_status;
    RAPIDSMP_MPI(MPI_Iprobe(MPI_ANY_SOURCE, tag, comm_, &msg_available, &probe_status));
    if (!msg_available) {
        return {nullptr, 0};
    }
    RAPIDSMP_EXPECTS(
        tag == probe_status.MPI_TAG || tag == MPI_ANY_TAG, "corrupt mpi tag"
    );
    MPI_Count size;
    RAPIDSMP_MPI(MPI_Get_elements_x(&probe_status, MPI_UINT8_T, &size));
    auto msg = std::make_unique<std::vector<uint8_t>>(size);  // TODO: uninitialize

    MPI_Status msg_status;
    RAPIDSMP_MPI(MPI_Recv(
        msg->data(),
        msg->size(),
        MPI_UINT8_T,
        probe_status.MPI_SOURCE,
        probe_status.MPI_TAG,
        comm_,
        &msg_status
    ));
    RAPIDSMP_MPI(MPI_Get_elements_x(&msg_status, MPI_UINT8_T, &size));
    RAPIDSMP_EXPECTS(
        static_cast<std::size_t>(size) == msg->size(),
        "incorrect size of the MPI_Recv message"
    );
    if (msg->size() > 2048) {  // TODO: use the actual eager threshold.
        log.warn(
            "block-receiving a messager larger than the normal ",
            "eager threshold (",
            msg->size(),
            " bytes)"
        );
    }
    return {std::move(msg), probe_status.MPI_SOURCE};
}

std::vector<std::size_t> MPI::test_some(
    std::vector<std::unique_ptr<Communicator::Future>> const& future_vector
) {
    std::vector<MPI_Request> reqs;
    for (auto const& future : future_vector) {
        auto mpi_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMP_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
        reqs.push_back(mpi_future->req_);
    }

    // Get completed requests as indices into `future_vector` (and `reqs`).
    std::vector<int> completed(reqs.size());
    {
        int num_completed{0};
        RAPIDSMP_MPI(MPI_Testsome(
            reqs.size(),
            reqs.data(),
            &num_completed,
            completed.data(),
            MPI_STATUSES_IGNORE
        ));
        completed.resize(num_completed);
    }
    return std::vector<std::size_t>(completed.begin(), completed.end());
}

std::vector<std::size_t> MPI::test_some(
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
        future_map
) {
    std::vector<MPI_Request> reqs;
    std::vector<std::size_t> key_reqs;
    reqs.reserve(future_map.size());
    key_reqs.reserve(future_map.size());
    for (auto const& [key, future] : future_map) {
        auto mpi_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMP_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
        reqs.push_back(mpi_future->req_);
        key_reqs.push_back(key);
    }

    // Get completed requests as indices into `key_reqs` (and `reqs`).
    std::vector<int> completed(reqs.size());
    {
        int num_completed{0};
        RAPIDSMP_MPI(MPI_Testsome(
            reqs.size(),
            reqs.data(),
            &num_completed,
            completed.data(),
            MPI_STATUSES_IGNORE
        ));
        completed.resize(num_completed);
    }

    std::vector<std::size_t> ret;
    ret.reserve(completed.size());
    for (int i : completed) {
        ret.push_back(key_reqs.at(i));
    }
    return ret;
}

std::unique_ptr<Buffer> MPI::get_gpu_data(std::unique_ptr<Communicator::Future> future) {
    auto mpi_future = dynamic_cast<Future*>(future.get());
    RAPIDSMP_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
    RAPIDSMP_EXPECTS(mpi_future->data_ != nullptr, "future has no data");
    return std::move(mpi_future->data_);
}

std::string MPI::str() const {
    int version, subversion;
    RAPIDSMP_MPI(MPI_Get_version(&version, &subversion));

    std::stringstream ss;
    ss << "MPI(rank=" << rank_ << ", nranks: " << nranks_ << ", mpi-version=" << version
       << "." << subversion << ")";
    return ss.str();
}

Rank UCXX::get_next_worker_rank() {
    if (rank_ != 0)
        throw std::runtime_error("This method can only be called by rank 0");
    return ++next_rank_;
}

}  // namespace rapidsmp
