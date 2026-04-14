/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <rapidsmpf/coll/gather_statistics.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::coll {

std::vector<std::shared_ptr<Statistics>> gather_statistics(
    std::shared_ptr<Communicator> const& comm,
    OpID op_id,
    std::shared_ptr<Statistics> const& stats,
    Rank root
) {
    RAPIDSMPF_EXPECTS(comm != nullptr, "Communicator must not be null");
    RAPIDSMPF_EXPECTS(stats != nullptr, "Statistics must not be null");

    auto const nranks = comm->nranks();
    auto const rank = comm->rank();
    Tag const tag{op_id, 0};

    if (nranks == 1) {
        return {};
    }

    if (rank != root) {
        // Serialize and send to root.
        auto serialized = stats->serialize();
        auto msg = std::make_unique<std::vector<std::uint8_t>>(std::move(serialized));
        auto future = comm->send(std::move(msg), root, tag);
        // Poll until send completes.
        while (!comm->test(future)) {
            std::this_thread::yield();
        }
        return {};
    }

    // Root: receive from all other ranks.
    // No ack/barrier is needed because the communicator guarantees no message
    // overtaking on the same (rank, tag) pair, so consecutive calls with the
    // same op_id cannot interfere.
    std::vector<std::shared_ptr<Statistics>> ret;
    ret.reserve(safe_cast<std::size_t>(nranks - 1));

    Rank received = 0;
    while (received < nranks - 1) {
        auto [msg, sender] = comm->recv_any(tag);
        if (msg) {
            ret.push_back(Statistics::deserialize(*msg));
            ++received;
        }
        std::this_thread::yield();
    }
    return ret;
}

}  // namespace rapidsmpf::coll
