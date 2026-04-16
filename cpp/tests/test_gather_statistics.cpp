/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <rapidsmpf/coll/gather_statistics.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "environment.hpp"

using namespace rapidsmpf;

extern Environment* GlobalEnvironment;

TEST(GatherStatisticsTest, Basic) {
    auto const& comm = GlobalEnvironment->comm_;
    auto const rank = comm->rank();
    auto const nranks = comm->nranks();

    auto stats = std::make_shared<Statistics>();
    stats->add_stat("x", safe_cast<double>(rank));

    auto others = coll::gather_statistics(comm, 0, stats);

    if (rank == 0) {
        // Root receives nranks - 1 Statistics from other ranks.
        ASSERT_EQ(others.size(), safe_cast<std::size_t>(nranks - 1));
        for (auto const& s : others) {
            EXPECT_TRUE(s->enabled());
            EXPECT_EQ(s->list_stat_names().size(), 1);
            // Each non-root rank sent stat "x" with its rank value.
            EXPECT_EQ(s->get_stat("x").count(), 1);
        }
        // Merge and verify the total.
        auto global = stats->merge(others);
        // Sum of ranks: 0 + 1 + 2 + ... + (nranks-1) = nranks*(nranks-1)/2
        double expected_sum = safe_cast<double>(nranks) * (nranks - 1) / 2.0;
        EXPECT_EQ(global->get_stat("x").value(), expected_sum);
        EXPECT_EQ(global->get_stat("x").count(), safe_cast<std::size_t>(nranks));
    } else {
        EXPECT_TRUE(others.empty());
    }

    GlobalEnvironment->barrier();
}

TEST(GatherStatisticsTest, SingleRank) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() != 1) {
        GTEST_SKIP() << "Test only meaningful with 1 rank";
    }

    auto stats = std::make_shared<Statistics>();
    stats->add_stat("x", 42.0);

    auto others = coll::gather_statistics(comm, 1, stats);
    EXPECT_TRUE(others.empty());
}

TEST(GatherStatisticsTest, DisjointStatNames) {
    auto const& comm = GlobalEnvironment->comm_;
    auto const rank = comm->rank();
    auto const nranks = comm->nranks();

    auto stats = std::make_shared<Statistics>();
    stats->add_stat("rank-" + std::to_string(rank), 1.0);

    auto others = coll::gather_statistics(comm, 2, stats);

    if (rank == 0) {
        ASSERT_EQ(others.size(), safe_cast<std::size_t>(nranks - 1));
        auto global = stats->merge(others);
        // Should have one stat per rank.
        EXPECT_EQ(global->list_stat_names().size(), safe_cast<std::size_t>(nranks));
        for (Rank r = 0; r < nranks; ++r) {
            EXPECT_EQ(global->get_stat("rank-" + std::to_string(r)).value(), 1.0);
        }
    } else {
        EXPECT_TRUE(others.empty());
    }

    GlobalEnvironment->barrier();
}
