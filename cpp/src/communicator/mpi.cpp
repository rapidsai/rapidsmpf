/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <memory>
#include <ranges>
#include <unordered_set>
#include <utility>

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

std::vector<int> mpi_testsome(std::vector<MPI_Request> const& reqs) {
    std::vector<int> indices(reqs.size());
    int num_completed{0};
    RAPIDSMPF_MPI(MPI_Testsome(
        reqs.size(),
        const_cast<MPI_Request*>(reqs.data()),
        &num_completed,
        indices.data(),
        MPI_STATUSES_IGNORE
    ));
    RAPIDSMPF_EXPECTS(
        num_completed != MPI_UNDEFINED, "Expected at least one active handle."
    );
    if (num_completed == 0) {
        return {};
    }

    indices.resize(static_cast<std::size_t>(num_completed));
    return indices;
}

bool mpi_testall(std::vector<MPI_Request> const& reqs) {
    int flag;
    RAPIDSMPF_MPI(MPI_Testall(
        static_cast<int>(reqs.size()),
        const_cast<MPI_Request*>(reqs.data()),
        &flag,
        MPI_STATUSES_IGNORE
    ));
    return static_cast<bool>(flag);
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
    RAPIDSMPF_EXPECTS(br != nullptr, "the BufferResource cannot be NULL");
    return send(br->move(std::move(msg)), rank, tag);
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

std::unique_ptr<Communicator::Future> MPI::send(
    std::unique_ptr<Buffer> msg, std::span<Rank> const ranks, Tag tag
) {
    RAPIDSMPF_EXPECTS(
        msg != nullptr && !ranks.empty(), "malformed arguments passed to batch send"
    );
    RAPIDSMPF_EXPECTS(
        msg->size <= std::numeric_limits<int>::max(),
        "send buffer size exceeds MPI max count"
    );
    std::vector<MPI_Request> reqs;
    reqs.reserve(ranks.size());
    for (auto rank : ranks) {
        MPI_Request req;
        RAPIDSMPF_MPI(
            MPI_Isend(msg->data(), msg->size, MPI_UINT8_T, rank, tag, comm_, &req)
        );
        reqs.emplace_back(req);
    }
    return std::make_unique<Future>(std::move(reqs), std::move(msg));
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
    std::vector<std::unique_ptr<Communicator::Future>> completed;

    std::vector<MPI_Request> singleton_reqs;
    singleton_reqs.reserve(future_vector.size());  // at most, all futures are singleton

    // partition the future_vector into singleton and multi-req futures
    auto [pivot, last] = std::ranges::partition(
        future_vector,
        [](Future* const mpi_future) {
            // if mpi_future is nullptr or multi-req future, it will be moved to the to
            // the end of the future_vector
            return mpi_future && mpi_future->size() == 1;
        },
        [&](auto& future) -> Future* {
            // project the future to a Future*: this will be called once for each future

            // if the future is a singleton future, add the request to the singleton_reqs
            // vector if the future is a multi-req future, test it immediately. If it's
            // completed, move it to the completed vector.
            auto mpi_future = dynamic_cast<Future*>(future.get());
            RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");

            if (mpi_future->size() == 1) {
                singleton_reqs.emplace_back(mpi_future->reqs_[0]);
            } else if (mpi_testall(mpi_future->reqs_)) {
                completed.emplace_back(std::move(future));
                return nullptr;
            }

            return mpi_future;
        }
    );

    // now, [future_vector.begin(), pivot) contains singleton futures
    // and [pivot, last) contains multi-req futures (with nullptr for completed ones)
    RAPIDSMPF_EXPECTS(
        singleton_reqs.size()
            == static_cast<size_t>(std::distance(future_vector.begin(), pivot)),
        "incorrect number of singleton requests"
    );
    // test the singleton requests
    auto completed_singleton_reqs = mpi_testsome(singleton_reqs);
    completed.reserve(completed.size() + completed_singleton_reqs.size());
    std::ranges::transform(
        completed_singleton_reqs.begin(),
        completed_singleton_reqs.end(),
        std::back_inserter(completed),
        [&](std::size_t i) { return std::move(future_vector[i]); }
    );

    // all completed futures should be nullptr in the future vector
    std::erase(future_vector, nullptr);

    return completed;
}

std::vector<std::size_t> MPI::test_some(
    std::unordered_map<std::size_t, std::unique_ptr<Communicator::Future>> const&
        future_map
) {
    std::vector<std::size_t> finished;
    std::vector<MPI_Request> singleton_reqs;
    std::vector<std::size_t> singleton_futures;

    // reserve for the most common case: all singleton futures
    singleton_reqs.reserve(future_map.size());
    singleton_futures.reserve(future_map.size());

    for (auto const& [key, future] : future_map) {
        auto mpi_future = dynamic_cast<Future const*>(future.get());
        RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
        if (mpi_future->size() == 1) {
            singleton_reqs.emplace_back(mpi_future->reqs_[0]);
            singleton_futures.emplace_back(key);
        } else {
            // test the multi-req futures immidiately
            auto indices = mpi_testsome(mpi_future->reqs_);
            if (indices.size() == mpi_future->size()) {
                finished.emplace_back(key);
            }
        }
    }

    // Test the singleton requests
    std::vector<int> completed = mpi_testsome(singleton_reqs);
    finished.reserve(finished.size() + completed.size());
    for (int i : completed) {
        finished.emplace_back(singleton_futures[static_cast<std::size_t>(i)]);
    }

    return finished;
}

std::unique_ptr<Buffer> MPI::wait(std::unique_ptr<Communicator::Future> future) {
    auto mpi_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
    RAPIDSMPF_MPI(MPI_Waitall(
        static_cast<int>(mpi_future->reqs_.size()),
        mpi_future->reqs_.data(),
        MPI_STATUS_IGNORE
    ));
    return std::move(mpi_future->data_);
}

std::unique_ptr<Buffer> MPI::get_gpu_data(std::unique_ptr<Communicator::Future> future) {
    auto mpi_future = dynamic_cast<Future*>(future.get());
    RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
    RAPIDSMPF_EXPECTS(mpi_future->data_ != nullptr, "future has no data");
    return std::move(mpi_future->data_);
}

bool MPI::test(Communicator::Future& future) {
    auto mpi_future = dynamic_cast<Future*>(&future);
    RAPIDSMPF_EXPECTS(mpi_future != nullptr, "future isn't a MPI::Future");
    return mpi_testall(mpi_future->reqs_);
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
