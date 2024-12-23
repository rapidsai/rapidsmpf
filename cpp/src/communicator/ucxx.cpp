/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/error.hpp>

namespace rapidsmp {

namespace mpi {
void init(int* argc, char*** argv) {
    int provided;

    // Initialize MPI with the desired level of thread support
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);

    RAPIDSMP_EXPECTS(
        provided == MPI_THREAD_MULTIPLE,
        "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
    );
}

void detail::check_mpi_error(int error_code, const char* file, int line) {
    if (error_code != MPI_SUCCESS) {
        std::array<char, MPI_MAX_ERROR_STRING> error_string;
        int error_length;
        MPI_Error_string(error_code, error_string.data(), &error_length);
        std::cerr << "MPI error at " << file << ":" << line << ": "
                  << std::string(error_string.data(), error_length) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }
}
}  // namespace mpi

namespace {
void check_mpi_thread_support() {
    int level;
    RAPIDSMP_MPI(MPI_Query_thread(&level));

    std::string level_str;
    switch (level) {
    case MPI_THREAD_SINGLE:
        level_str = "MPI_THREAD_SINGLE";
        break;
    case MPI_THREAD_FUNNELED:
        level_str = "MPI_THREAD_FUNNELED";
        break;
    case MPI_THREAD_SERIALIZED:
        level_str = "MPI_THREAD_SERIALIZED";
        break;
    case MPI_THREAD_MULTIPLE:
        level_str = "MPI_THREAD_MULTIPLE";
        break;
    default:
        throw std::logic_error("MPI_Query_thread(): unknown thread level support");
    }
    RAPIDSMP_EXPECTS(
        level == MPI_THREAD_MULTIPLE,
        "MPI thread level support " + level_str
            + " isn't sufficient, need MPI_THREAD_MULTIPLE"
    );
}
}  // namespace

static void listener_callback(ucp_conn_request_h conn_request, void* arg) {
    auto listener_container = reinterpret_cast<ListenerContainer*>(arg);

    ucp_conn_request_attr_t attr{};
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    auto status = ucp_conn_request_query(connRequest, &attr)
        :  // TODO: Log if creating endpoint failed?
          if (status != UCS_OK) return;

    auto endpoint =
        listener_container->listener_->createEndpointFromConnRequest(connRequest, true);
    endpoints_[endpoint->getHandle()] = endpoint;

    if (listener_container->root) {
        // TODO: Reuse receive_rank_callback_info
        ucxx::AmReceiverCallbackInfo receive_rank_callback_info("rapidsmp", 0);
        // TODO: Ensure nextRank remains alive until request completes
        auto next_rank = get_next_worker_rank();
        auto req = ep->amSend(
            &next_rank,
            sizeof(next_rank),
            ucxx::BufferType::Host,
            receive_rank_callback_info
        );
    }
}

UCXX::UCXX(std::shared_ptr<ucxx::Worker> worker, bool root, std::uint32_t nranks)
    : worker_{worker},
      rank_{Rank(-1)},
      nranks_{nranks},
      endpoints_{std::make_shared<EndpointsMap>},
      logger_{this} {
    // int rank;
    // int nranks;
    // RAPIDSMP_MPI(MPI_Comm_rank(comm_, &rank));
    // RAPIDSMP_MPI(MPI_Comm_size(comm_, &nranks));
    // rank_ = rank;
    // nranks_ = nranks;
    // check_mpi_thread_support();

    if (worker_ == nullptr) {
        auto context = ucxx::createContext({{}}, ucxx::Context::defaultFeatureFlags);
        worker_ = context->createWorker(false);
    }

    // Create listener
    listener_ = worker_->createListener(0, listener_callback, listener_container_.get());
    listener_container_.listener_ = listener_;
    listener_container_.endpoints_ = endpoints_;

    ucxx::AmReceiverCallbackInfo receive_rank_callback_info("rapidsmp", 0);
    auto receive_rank = ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h ep) {
            rank_ = *reinterpret_cast<Rank*>(req->getRecvBuffer()->data());
        }
    )

        ucxx::AmReceiverCallbackInfo endpoint_registration_callback_info("rapidsmp", 1);
    auto endpoint_registration_callback = ucxx::AmReceiverCallbackType(
        [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h ep) {
            auto rank = *reinterpret_cast<Rank*>(req->getRecvBuffer()->data());
            rank_to_endpoint_[rank] = ep;
        }
    )

        ucxx::AmReceiverCallbackInfo receive_listener_address_callback_info(
            "rapidsmp", 2
        );
    auto receive_listener_address =
        ucxx::AmReceiverCallbackType(
            [this](std::shared_ptr<ucxx::Request> req, ucp_ep_h ep) {
                auto listener_address =
                    reinterpret_cast<ListenerAddress*>(req->getRecvBuffer()->data());
                rank_to_listener_address_[listener_address->rank] = ListenerAddress {
                    .host = listener_address->host;
                    .port = listener_address->host;
                    .rank = listener_address->rank;
                }
            }
        )

            worker_->registerAmReceiverCallback(
                endpoint_registration_callback_info, endpoint_registration_callback
            );
    worker_->registerAmReceiverCallback(receive_rank_callback_info, receive_rank);
    worker_->registerAmReceiverCallback(
        receive_listener_address_callback_info, receive_listener_address
    );

    if (root) {
        rank_ = Rank(0);
    } else {
        // Connect to root
        auto endpoint = worker_->createEndpointFromHostname("localhost", port, true);
        rank_to_endpoint_[Rank(0)] = endpoint->getHandle();
        endpoints_[endpoint->getHandle()] = endpoint;

        // Get my rank
        while (rank_ == Rank(-1)) {
            // TODO: progress
        }

        // Inform listener address
        ListenerAddress listener_address = ListenerAddress{
            .host = "localhost", .port = listener_->getPort(), .rank = rank_
        };
        endpoint->amSend(
            static_cast<void*>(&listener_address),
            sizeof(listener_address),
            ucxx::Buffer::Host,
            receive_listener_address_callback_info
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
}  // namespace rapidsmp
