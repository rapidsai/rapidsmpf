/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <rapidsmpf/communicator/communicator.hpp>

namespace rapidsmpf {

class BufferResource;

/**
 * @brief Kind of filesystem that holds a rank's spill directory.
 */
enum class FsType : std::uint8_t {
    NONE,  ///< The rank has no `DiskResource`.
    LOCAL,  ///< The spill directory is on a node-local filesystem.
    NETWORK,  ///< The spill directory is on a network filesystem (NFS, Lustre, ...).
};

/**
 * @brief Get the name of a filesystem kind.
 *
 * @param type The filesystem kind.
 * @return "none", "local" or "network".
 */
[[nodiscard]] std::string_view to_string(FsType type) noexcept;

/**
 * @brief Filesystem kind and mount fingerprint of a rank's spill directory.
 */
struct SpillDirInfo {
    /// Fingerprint of the network mount holding the spill directory: a hash of the
    /// filesystem type, the mount source and the mount point. Two ranks with the same
    /// non-zero value see the same files at the same absolute paths. Zero unless
    /// `fs_type == FsType::NETWORK`.
    std::uint64_t mount_id;

    /// Kind of filesystem holding the spill directory.
    FsType fs_type;
};

/**
 * @brief Per-rank information exchanged when building a `RankTopology`.
 *
 * The struct is flat and trivially copyable so that it can be exchanged as raw bytes.
 * All ranks must therefore run the same build on the same ABI.
 */
struct RankInfo {
    /// Identifies the machine: a hash of the hostname and the kernel boot id.
    std::uint64_t node_id;

    /// Mount-namespace id (inode of `/proc/self/ns/mnt`). Processes on the same node but
    /// in different containers usually have different mount namespaces, and may see
    /// different files at the same path.
    std::uint64_t mount_ns;

    /// Filesystem kind and mount fingerprint of the spill directory.
    SpillDirInfo spill_dir;

    /// Hostname, NUL-terminated (`HOST_NAME_MAX + 1` bytes). Used for diagnostics only.
    std::array<char, 65> hostname;

    /// The rank this information describes.
    Rank rank;

    /// CUDA device ordinal of the rank (`cudaGetDevice()`), or -1 if it has none. The
    /// ordinal is relative to the rank's `CUDA_VISIBLE_DEVICES`.
    std::int32_t gpu_device;

    /**
     * @brief Get the hostname.
     *
     * @return A view of the NUL-terminated hostname.
     */
    [[nodiscard]] std::string_view hostname_view() const noexcept;

    /**
     * @brief Whether the rank has a `DiskResource`.
     *
     * @return True if the rank can spill to disk.
     */
    [[nodiscard]] constexpr bool has_disk() const noexcept {
        return spill_dir.fs_type != FsType::NONE;
    }

    /**
     * @brief Whether the rank's spill directory is on a network filesystem.
     *
     * @return True if `spill_dir.fs_type == FsType::NETWORK`.
     */
    [[nodiscard]] constexpr bool spill_is_network_fs() const noexcept {
        return spill_dir.fs_type == FsType::NETWORK;
    }
};

static_assert(std::is_trivially_copyable_v<RankInfo>);
static_assert(std::is_standard_layout_v<RankInfo>);

/**
 * @brief How the ranks of a communicator are placed relative to each other.
 *
 * Every rank holds the `RankInfo` of all ranks. From it, a rank can find which ranks
 * share its node, and whether a rank can read the spill files written by another
 * rank. The latter is the basis for handing disk buffers over by path instead of
 * sending their bytes.
 *
 * Spill-file visibility is decided by a rule, not by testing files: rank `reader` can
 * read the spill files of rank `owner` if `owner` has a `DiskResource` and either
 *  - both ranks are on the same node and in the same mount namespace, or
 *  - both spill directories are on the same network mount (same filesystem type,
 *    source and mount point).
 *
 * The rule can be wrong, for example when two mount namespaces bind-mount the same
 * directory. Users of the result must be able to fall back to sending the data.
 */
class RankTopology {
  public:
    /**
     * @brief Build the topology of a communicator.
     *
     * Collective: every rank of @p comm must call this. It runs one allgather with the
     * reserved OpID `detail::rank_topology_op_id`, so it must not run concurrently with
     * another `create()` on the same communicator.
     *
     * @param comm The communicator.
     * @param br Buffer resource. Its `DiskResource`, if any, describes this rank's
     * spill directory.
     * @return The topology, identical on every rank except for which rank is local.
     */
    [[nodiscard]] static std::shared_ptr<RankTopology> create(
        std::shared_ptr<Communicator> const& comm,
        std::shared_ptr<BufferResource> const& br
    );

    /**
     * @brief Construct a topology from already-gathered information.
     *
     * @param rank The calling rank.
     * @param ranks Information of every rank, where `ranks[r].rank == r`.
     *
     * @throws std::invalid_argument If @p ranks is not indexed by rank.
     * @throws std::out_of_range If @p rank is not a valid rank.
     */
    RankTopology(Rank rank, std::vector<RankInfo> ranks);

    /**
     * @brief Get the information of the calling rank.
     *
     * @return The information of the calling rank.
     */
    [[nodiscard]] constexpr RankInfo const& local() const noexcept {
        return ranks_[static_cast<std::size_t>(rank_)];
    }

    /**
     * @brief Get the information of a rank.
     *
     * @param r The rank.
     * @return The information of @p r.
     *
     * @throws std::out_of_range If @p r is not a valid rank.
     */
    [[nodiscard]] constexpr RankInfo const& info(Rank r) const {
        // A negative rank wraps to a huge index, which `at()` also rejects.
        return ranks_.at(static_cast<std::size_t>(r));
    }

    /**
     * @brief Get the information of all ranks.
     *
     * @return The information of every rank, indexed by rank.
     */
    [[nodiscard]] std::span<RankInfo const> ranks() const noexcept {
        return ranks_;
    }

    /**
     * @brief Get the ranks on the same node as the calling rank.
     *
     * @return The ranks in increasing order, including the calling rank.
     */
    [[nodiscard]] std::span<Rank const> node_local_ranks() const noexcept {
        return node_local_ranks_;
    }

    /**
     * @brief Get the index of the calling rank among the ranks on its node.
     *
     * @return The position of the calling rank in `node_local_ranks()`.
     */
    [[nodiscard]] std::size_t local_rank_idx() const noexcept {
        return local_rank_idx_;
    }

    /**
     * @brief Whether two ranks run on the same node.
     *
     * @param a A rank.
     * @param b A rank.
     * @return True if @p a and @p b run on the same node.
     *
     * @throws std::out_of_range If either rank is not valid.
     */
    [[nodiscard]] constexpr bool same_node(Rank a, Rank b) const {
        return info(a).node_id == info(b).node_id;
    }

    /**
     * @brief Whether a rank can read the spill files written by another rank.
     *
     * @param reader The rank that would open the file.
     * @param owner The rank whose `DiskResource` wrote the file.
     * @return True if @p reader is expected to see the files of @p owner at the same
     * absolute paths.
     *
     * @throws std::out_of_range If either rank is not valid.
     */
    [[nodiscard]] constexpr bool can_read_spill_files(Rank reader, Rank owner) const {
        auto const& r = info(reader);
        auto const& o = info(owner);
        if (!o.has_disk()) {
            return false;
        }
        bool const same_fs_view = (r.node_id == o.node_id) && (r.mount_ns == o.mount_ns);
        bool const same_network_mount = r.spill_is_network_fs() && o.spill_is_network_fs()
                                        && r.spill_dir.mount_id == o.spill_dir.mount_id;
        return same_fs_view || same_network_mount;
    }

    /**
     * @brief Get a description of the topology.
     *
     * @return A string describing the topology.
     */
    [[nodiscard]] std::string str() const;

  private:
    Rank rank_;
    std::vector<RankInfo> ranks_;
    std::vector<Rank> node_local_ranks_;
    std::size_t local_rank_idx_{0};
};

}  // namespace rapidsmpf
