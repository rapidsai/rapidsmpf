/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <utility>
#include <variant>

#include <mpi.h>

#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

namespace mpi {
void init(int* argc, char*** argv) {
    if (!is_initialized()) {
        int provided;

        // Initialize MPI with the desired level of thread support
        RAPIDSMPF_MPI(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided));

        RAPIDSMPF_EXPECTS(
            provided == MPI_THREAD_MULTIPLE,
            "didn't get the requested thread level support: MPI_THREAD_MULTIPLE"
        );
    }

    // Check if max MPI TAG can accommodate the OpID + TagPrefixT
    int flag;
    int32_t* max_tag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag, &flag);
    RAPIDSMPF_EXPECTS(flag, "Unable to get the MPI_TAG_UB attr");

    RAPIDSMPF_EXPECTS(
        (*max_tag) >= Tag::max_value(),
        "MPI_TAG_UB(" + std::to_string(*max_tag)
            + ") is unable to accommodate the required max tag("
            + std::to_string(Tag::max_value()) + ")"
    );
}

bool is_initialized() {
    int flag;
    RAPIDSMPF_MPI(MPI_Initialized(&flag));
    return flag;
}

void detail::check_mpi_error(int error_code, const char* file, int line) {
    if (error_code != MPI_SUCCESS) {
        std::array<char, MPI_MAX_ERROR_STRING> error_string;
        int error_length;
        MPI_Error_string(error_code, error_string.data(), &error_length);
        std::cerr << "MPI error at " << file << ":" << line << ": "
                  << std::string(
                         error_string.data(), static_cast<std::size_t>(error_length)
                     )
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error_code);
    }
}
}  // namespace mpi

namespace {
void check_mpi_thread_support() {
    int level;
    RAPIDSMPF_MPI(MPI_Query_thread(&level));

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
    RAPIDSMPF_EXPECTS(
        level == MPI_THREAD_MULTIPLE,
        "MPI thread level support " + level_str
            + " isn't sufficient, need MPI_THREAD_MULTIPLE"
    );
}
}  // namespace

MPI::MPI(MPI_Comm comm, config::Options options)
    : comm_{comm}, logger_{this, std::move(options)} {
    int rank;
    int nranks;
    RAPIDSMPF_MPI(MPI_Comm_rank(comm_, &rank));
    RAPIDSMPF_MPI(MPI_Comm_size(comm_, &nranks));
    rank_ = rank;
    nranks_ = nranks;
    check_mpi_thread_support();
}

std::unique_ptr<Communicator::Future> MPI::send(
    std::unique_ptr<std::vector<uint8_t>> msg, Rank rank, Tag tag, BufferResource* br
) {
    RAPIDSMPF_EXPECTS(
        msg->size() <= std::numeric_limits<int>::max(),
        "send buffer size exceeds MPI max count"
    );
    MPI_Request req;
    RAPIDSMPF_MPI(
        MPI_Isend(msg->data(), msg->size(), MPI_UINT8_T, rank, tag, comm_, &req)
    );
    return std::make_unique<Future>(req, br->move(std::move(msg)));
}

std::unique_ptr<Communicator::Future> MPI::send(
    std::unique_ptr<Buffer> msg, Rank rank, Tag tag
) {
    if (!msg->is_ready()) {
        logger().warn("msg is not ready. This is irrecoverable, terminating.");
        std::terminate();
    }
    RAPIDSMPF_EXPECTS(
        msg->size <= std::numeric_limits<int>::max(),
        "send buffer size exceeds MPI max count"
    );
    MPI_Request req;
    RAPIDSMPF_MPI(MPI_Isend(msg->data(), msg->size, MPI_UINT8_T, rank, tag, comm_, &req));
    return std::make_unique<Future>(req, std::move(msg));
}

std::vector<std::unique_ptr<Communicator::Future>> MPI::send(
    std::unique_ptr<Buffer> msg, std::span<Rank> const destinations, Tag tag
) {
    if (!msg->is_ready()) {
        logger().warn("msg is not ready. This is irrecoverable, terminating.");
        std::terminate();
    }
    RAPIDSMPF_EXPECTS(
        msg->size <= std::numeric_limits<int>::max(),
        "send buffer size exceeds MPI max count"
    );
    std::vector<std::unique_ptr<Communicator::Future>> futures;
    futures.reserve(destinations.size());
    auto buf = std::shared_ptr<Buffer>(msg.release(), msg.get_deleter());
    std::ranges::transform(
        destinations, std::back_inserter(futures), [&](Rank const dest) {
            MPI_Request req;
            RAPIDSMPF_MPI(
                MPI_Isend(buf->data(), buf->size, MPI_UINT8_T, dest, tag, comm_, &req)
            );
            return std::make_unique<Future>(req, buf);
        }
    );
    return futures;
}

std::unique_ptr<Communicator::Future> MPI::recv(
    Rank rank, Tag tag, std::unique_ptr<Buffer> recv_buffer
) {
    if (!recv_buffer->is_ready()) {
        logger().warn("recv_buffer is not ready. This is irrecoverable, terminating.");
        std::terminate();
    }
    RAPIDSMPF_EXPECTS(
        recv_buffer->size <= std::numeric_limits<int>::max(),
        "recv buffer size exceeds MPI max count"
    );
    MPI_Request req;
    RAPIDSMPF_MPI(MPI_Irecv(
        recv_buffer->data(), recv_buffer->size, MPI_UINT8_T, rank, tag, comm_, &req
    ));
    return std::make_unique<Future>(req, std::move(recv_buffer));
}

std::pair<std::unique_ptr<std::vector<uint8_t>>, Rank> MPI::recv_any(Tag tag) {
    int msg_available;
    MPI_Status probe_status;
    MPI_Message matched_msg;
    RAPIDSMPF_MPI(MPI_Improbe(
        MPI_ANY_SOURCE, tag, comm_, &msg_available, &matched_msg, &probe_status
    ));
    if (!msg_available) {
        return {nullptr, 0};
    }
    RAPIDSMPF_EXPECTS(
        tag == probe_status.MPI_TAG || tag == MPI_ANY_TAG, "corrupt mpi tag"
    );
    MPI_Count size;
    RAPIDSMPF_MPI(MPI_Get_elements_x(&probe_status, MPI_UINT8_T, &size));
    RAPIDSMPF_EXPECTS(
        size <= std::numeric_limits<int>::max(), "recv buffer size exceeds MPI max count"
    );
    auto msg = std::make_unique<std::vector<uint8_t>>(size);  // TODO: uninitialize

    MPI_Status msg_status;
    RAPIDSMPF_MPI(
        MPI_Mrecv(msg->data(), msg->size(), MPI_UINT8_T, &matched_msg, &msg_status)
    );
    RAPIDSMPF_MPI(MPI_Get_elements_x(&msg_status, MPI_UINT8_T, &size));
    RAPIDSMPF_EXPECTS(
        static_cast<std::size_t>(size) == msg->size(),
        "incorrect size of the MPI_Recv message"
    );
    return {std::move(msg), probe_status.MPI_SOURCE};
}

std::vector<std::unique_ptr<Communicator::Future>> MPI::test_some(
    std::vector<std::unique_ptr<Communicator::Future>>& future_vector
) {
    if (future_vector.empty()) {
        return {};
    }
    std::vector<MPI_Request> reqs;
    reqs.reserve(future_vector.size());
    for (auto const& future : future_vector) {
        auto mpi_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
        reqs.push_back(mpi_future->req_);
    }

    // Get completed requests as indices into `future_vector` (and `reqs`).
    std::vector<int> indices(reqs.size());
    int num_completed{0};
    RAPIDSMPF_MPI(MPI_Testsome(
        reqs.size(), reqs.data(), &num_completed, indices.data(), MPI_STATUSES_IGNORE
    ));
    RAPIDSMPF_EXPECTS(
        num_completed != MPI_UNDEFINED, "Expected at least one active handle."
    );
    if (num_completed == 0) {
        return {};
    }
    std::vector<std::unique_ptr<Communicator::Future>> completed;
    completed.reserve(static_cast<std::size_t>(num_completed));
    std::ranges::transform(
        indices.begin(),
        indices.begin() + num_completed,
        std::back_inserter(completed),
        [&](std::size_t i) { return std::move(future_vector[i]); }
    );
    std::erase(future_vector, nullptr);
    return completed;
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
        RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
        reqs.push_back(mpi_future->req_);
        key_reqs.push_back(key);
    }

    // Get completed requests as indices into `key_reqs` (and `reqs`).
    std::vector<int> completed(reqs.size());
    {
        int num_completed{0};
        RAPIDSMPF_MPI(MPI_Testsome(
            reqs.size(),
            reqs.data(),
            &num_completed,
            completed.data(),
            MPI_STATUSES_IGNORE
        ));
        completed.resize(static_cast<std::size_t>(num_completed));
    }

    std::vector<std::size_t> ret;
    ret.reserve(completed.size());
    for (int i : completed) {
        ret.push_back(key_reqs.at(static_cast<std::size_t>(i)));
    }
    return ret;
}

std::unique_ptr<Buffer> MPI::wait(std::unique_ptr<Communicator::Future> future) {
    auto mpi_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
    RAPIDSMPF_MPI(MPI_Wait(&mpi_future->req_, MPI_STATUS_IGNORE));
    RAPIDSMPF_EXPECTS(
        std::holds_alternative<std::unique_ptr<Buffer>>(mpi_future->data_),
        "Can't wait on future holding shared pointer"
    );
    return std::move(std::get<std::unique_ptr<Buffer>>(mpi_future->data_));
}

std::unique_ptr<Buffer> MPI::get_gpu_data(std::unique_ptr<Communicator::Future> future) {
    auto mpi_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
    RAPIDSMPF_EXPECTS(
        std::holds_alternative<std::unique_ptr<Buffer>>(mpi_future->data_),
        "Can't get_gpu_data on future holding shared pointer"
    );
    auto result = std::move(std::get<std::unique_ptr<Buffer>>(mpi_future->data_));
    RAPIDSMPF_EXPECTS(result != nullptr, "future has no data");
    return result;
}

std::string MPI::str() const {
    int version, subversion;
    RAPIDSMPF_MPI(MPI_Get_version(&version, &subversion));

    std::stringstream ss;
    ss << "MPI(rank=" << rank_ << ", nranks: " << nranks_ << ", mpi-version=" << version
       << "." << subversion << ")";
    return ss.str();
}
}  // namespace rapidsmpf
