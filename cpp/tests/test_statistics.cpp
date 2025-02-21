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


#include <gtest/gtest.h>

#include <rapidsmp/communicator/mpi.hpp>
#include <rapidsmp/statistics.hpp>

using namespace rapidsmp;

TEST(Statistics, Disabled) {
    rapidsmp::Statistics stats;
    EXPECT_FALSE(stats.enabled());

    // Disabed statistics is a no-op.
    EXPECT_EQ(stats.add_peer_comm(1, 10), 0);
    EXPECT_EQ(stats.get_peer_stats(42), rapidsmp::Statistics::PeerStats{});
}

TEST(Statistics, Communication) {
    std::shared_ptr<rapidsmp::Communicator> comm =
        std::make_shared<rapidsmp::MPI>(MPI_COMM_WORLD);
    rapidsmp::Statistics stats(comm);
    EXPECT_TRUE(stats.enabled());

    // Invalid rank.
    EXPECT_THROW(stats.add_peer_comm(3, 10), std::out_of_range);
    EXPECT_THROW(stats.get_peer_stats(3), std::out_of_range);

    EXPECT_EQ(stats.add_peer_comm(1, 10), 10);
    EXPECT_EQ(stats.get_peer_stats(0).comm_count, 0);
    EXPECT_EQ(stats.get_peer_stats(0).comm_nbytes, 0);
    EXPECT_EQ(stats.get_peer_stats(1).comm_count, 1);
    EXPECT_EQ(stats.get_peer_stats(1).comm_nbytes, 10);
    EXPECT_EQ(stats.add_peer_comm(1, 10), 20);
    EXPECT_EQ(stats.get_peer_stats(1).comm_count, 2);
    EXPECT_EQ(stats.get_peer_stats(1).comm_nbytes, 20);
}
