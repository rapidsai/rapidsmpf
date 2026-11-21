/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/rank_topology.hpp>

#include "environment.hpp"
#include "utils.hpp"

using namespace rapidsmpf;

extern Environment* GlobalEnvironment;

namespace {

RankInfo make_info(
    Rank rank,
    std::uint64_t node_id,
    FsType fs_type = FsType::LOCAL,
    std::uint64_t mount_id = 0,
    std::uint64_t mount_ns = 1
) {
    RankInfo info{};
    info.rank = rank;
    info.node_id = node_id;
    info.mount_ns = mount_ns;
    info.spill_dir.fs_type = fs_type;
    info.spill_dir.mount_id = mount_id;
    info.gpu_device = 0;
    return info;
}

/// Whether @p reader can read the spill files of @p owner (the local rank), in a two-rank
/// topology.
bool visible(RankInfo reader, RankInfo owner) {
    reader.rank = 0;
    owner.rank = 1;
    return RankTopology{1, {reader, owner}}.can_read_spill_files(0);
}

}  // namespace

TEST(RankTopologyLocal, SpillVisibilityRule) {
    auto const same_node = make_info(0, 7);
    auto const peer = make_info(1, 7);
    EXPECT_TRUE(visible(same_node, peer));

    // Owner without a disk resource has no files to read.
    EXPECT_FALSE(visible(same_node, make_info(1, 7, FsType::NONE)));

    // Same node, different mount namespace.
    EXPECT_FALSE(visible(same_node, make_info(1, 7, FsType::LOCAL, 0, 2)));

    // Different nodes, local filesystems.
    EXPECT_FALSE(visible(make_info(0, 7), make_info(1, 8)));

    // Different nodes on the same network mount, regardless of mount namespace.
    auto const nfs_a = make_info(0, 7, FsType::NETWORK, 42, 1);
    auto const nfs_b = make_info(1, 8, FsType::NETWORK, 42, 5);
    EXPECT_TRUE(visible(nfs_a, nfs_b));
    EXPECT_TRUE(visible(nfs_b, nfs_a));

    // Different network mounts.
    EXPECT_FALSE(visible(nfs_a, make_info(1, 8, FsType::NETWORK, 43, 1)));

    // Only one side on the network mount.
    EXPECT_FALSE(visible(make_info(0, 7), nfs_b));
}

TEST(RankTopologyLocal, ConstructorValidatesRanks) {
    EXPECT_THROW(RankTopology(0, {make_info(1, 7)}), std::invalid_argument);
    EXPECT_THROW(RankTopology(1, {make_info(0, 7)}), std::out_of_range);
    EXPECT_THROW(RankTopology(-1, {make_info(0, 7)}), std::out_of_range);
}

TEST(RankTopologyLocal, GroupsRanksByNode) {
    RankTopology const topo{
        2, {make_info(0, 7), make_info(1, 8), make_info(2, 7), make_info(3, 8)}
    };
    EXPECT_EQ(topo.local().rank, 2);
    EXPECT_EQ(topo.ranks().size(), std::size_t{4});
    EXPECT_EQ(
        std::vector<Rank>(topo.node_local_ranks().begin(), topo.node_local_ranks().end()),
        (std::vector<Rank>{0, 2})
    );
    EXPECT_EQ(topo.local_rank_idx(), std::size_t{1});
    EXPECT_TRUE(topo.same_node(1, 3));
    EXPECT_FALSE(topo.same_node(0, 1));
    EXPECT_TRUE(topo.can_read_spill_files(0));
    EXPECT_FALSE(topo.can_read_spill_files(1));
    EXPECT_THROW(std::ignore = topo.can_read_spill_files(4), std::out_of_range);
    EXPECT_THROW(std::ignore = topo.info(4), std::out_of_range);
    EXPECT_THROW(std::ignore = topo.same_node(0, -1), std::out_of_range);
    EXPECT_NE(topo.str().find("node_local_ranks=[0, 2]"), std::string::npos);
}

TEST(RankTopology, CreateWithoutDisk) {
    auto const& comm = GlobalEnvironment->comm_;
    auto br = create_br();
    auto const topo = RankTopology::create(comm, br);

    ASSERT_EQ(topo->ranks().size(), static_cast<std::size_t>(comm->nranks()));
    EXPECT_EQ(topo->local().rank, comm->rank());
    EXPECT_EQ(
        std::memcmp(&topo->info(comm->rank()), &topo->local(), sizeof(RankInfo)), 0
    );

    // Information collected about the calling process.
    auto const& local = topo->local();
    std::array<char, 256> hostname{};
    ASSERT_EQ(::gethostname(hostname.data(), hostname.size() - 1), 0);
    EXPECT_EQ(local.hostname_view(), std::string_view{hostname.data()});
    EXPECT_NE(local.node_id, 0);
    EXPECT_NE(local.mount_ns, 0);
    EXPECT_GE(local.gpu_device, 0);
    EXPECT_EQ(local.spill_dir.fs_type, FsType::NONE);
    EXPECT_EQ(local.spill_dir.mount_id, 0);

    for (Rank r = 0; r < comm->nranks(); ++r) {
        EXPECT_EQ(topo->info(r).rank, r);
        EXPECT_FALSE(topo->info(r).has_disk());
        for (Rank o = 0; o < comm->nranks(); ++o) {
            EXPECT_EQ(topo->same_node(r, o), topo->same_node(o, r));
        }
        EXPECT_FALSE(topo->can_read_spill_files(r));
    }

    // All test ranks run on this machine.
    auto const local_ranks = topo->node_local_ranks();
    EXPECT_EQ(local_ranks.size(), static_cast<std::size_t>(comm->nranks()));
    EXPECT_TRUE(std::ranges::is_sorted(local_ranks));
    EXPECT_EQ(local_ranks[topo->local_rank_idx()], comm->rank());
}

TEST(RankTopology, CreateWithDisk) {
    auto const& comm = GlobalEnvironment->comm_;
    TempDir dir;  // Declared before `br` so it outlives the buffer resource.
    auto br = create_br(dir.path());
    auto const topo = RankTopology::create(comm, br);

    // The test temp dir may itself be on a network filesystem.
    auto const& local = topo->local();
    if (local.spill_dir.fs_type == FsType::LOCAL) {
        EXPECT_EQ(local.spill_dir.mount_id, 0);
    } else {
        EXPECT_EQ(local.spill_dir.fs_type, FsType::NETWORK);
        EXPECT_NE(local.spill_dir.mount_id, 0);
    }

    for (Rank r = 0; r < comm->nranks(); ++r) {
        EXPECT_TRUE(topo->info(r).has_disk());
        // Same node and, in these tests, the same mount namespace.
        EXPECT_TRUE(topo->can_read_spill_files(r));
    }
}

TEST(RankTopology, CreateTwiceReusesOpID) {
    auto const& comm = GlobalEnvironment->comm_;
    auto br = create_br();
    auto const first = RankTopology::create(comm, br);
    auto const second = RankTopology::create(comm, br);
    ASSERT_EQ(first->ranks().size(), second->ranks().size());
    for (Rank r = 0; r < comm->nranks(); ++r) {
        EXPECT_EQ(first->info(r).node_id, second->info(r).node_id);
    }
}
