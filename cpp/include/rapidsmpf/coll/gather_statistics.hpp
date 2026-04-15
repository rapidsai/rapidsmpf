/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf::coll {

/**
 * @brief Gather statistics from all non-root ranks to the root rank.
 *
 * Non-root ranks serialize and send their statistics to the root rank. On the
 * root rank the @p stats argument is ignored and the return value contains the
 * deserialized statistics from every other rank. On non-root ranks the return
 * value is an empty vector.
 *
 * This is a blocking collective: all ranks must call this function.
 *
 * @note The current implementation is not optimized for performance and should
 * not be called on a critical path.
 *
 * @note It is safe to reuse the @p op_id as soon as this function has returned.
 *
 * Example usage:
 * @code{.cpp}
 * Rank root = 0;
 * auto others = coll::gather_statistics(comm, op_id, stats, root);
 * if (comm->rank() == root) {
 *     auto global = stats->merge(others);
 *     std::cout << global->report();
 * }
 * @endcode
 *
 * @param comm The communicator.
 * @param op_id Operation ID for tag disambiguation. Must be the same on all ranks.
 * @param stats The local statistics to send (ignored on root). Must not be null.
 * @param root The root rank that collects the statistics (default 0).
 * @note The gathered Statistics objects contain only stats, no memory records
 * or formatters.
 *
 * @return On root: a vector of `nranks - 1` deserialized Statistics from all
 * non-root ranks. On non-root ranks: an empty vector.
 */
[[nodiscard]] std::vector<std::shared_ptr<Statistics>> gather_statistics(
    std::shared_ptr<Communicator> const& comm,
    OpID op_id,
    std::shared_ptr<Statistics> const& stats = Statistics::disabled(),
    Rank root = 0
);

}  // namespace rapidsmpf::coll
