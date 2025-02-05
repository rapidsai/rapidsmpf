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

    // Check if max MPI TAG can accommodate the OpID + TagPrefixT
    int flag;
    int32_t* max_tag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag);
    RAPIDSMP_EXPECTS(flag, "Unable to get the MPI_TAG_UB attr");

    // RAPIDSMP_EXPECTS(
    //     (*max_tag) >= Tag::max_value(),
    //     "MPI_TAG_UB(" + std::to_string(*max_tag)
    //         + ") is unable to accommodate the required max tag("
    //         + std::to_string(Tag::max_value()) + ")"
    // );
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

MPI::MPI(MPI_Comm comm) : comm_{comm}, logger_{this} {
    int rank;
    int nranks;
    RAPIDSMP_MPI(MPI_Comm_rank(comm_, &rank));
    RAPIDSMP_MPI(MPI_Comm_size(comm_, &nranks));
    rank_ = rank;
    nranks_ = nranks;
    check_mpi_thread_support();
}

std::unique_ptr<Communicator::Future> MPI::send(
    std::unique_ptr<std::vector<uint8_t>> msg,
    Rank rank,
    Tag tag,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    RAPIDSMP_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
    MPI_Request req;
    RAPIDSMP_MPI(MPI_Isend(msg->data(), msg->size(), MPI_UINT8_T, rank, tag, comm_, &req)
    );
    return std::make_unique<Future>(req, br->move(std::move(msg), stream));
}

std::unique_ptr<Communicator::Future> MPI::send(
    std::unique_ptr<Buffer> msg, Rank rank, Tag tag, rmm::cuda_stream_view stream
) {
    MPI_Request req;
    RAPIDSMP_MPI(MPI_Isend(msg->data(), msg->size, MPI_UINT8_T, rank, tag, comm_, &req));
    return std::make_unique<Future>(req, std::move(msg));
}

std::unique_ptr<Communicator::Future> MPI::recv(
    Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer, rmm::cuda_stream_view stream
) {
    MPI_Request req;
    RAPIDSMP_MPI(MPI_Irecv(
        recv_buffer->data(), recv_buffer->size, MPI_UINT8_T, rank, tag, comm_, &req
    ));
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> MPI::recv_any(Tag tag) {
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
