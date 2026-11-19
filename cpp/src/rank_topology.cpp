/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rapidsmpf/coll/allgather.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/rank_topology.hpp>
#include <rapidsmpf/system_info.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf {

std::string_view to_string(FsType type) noexcept {
    switch (type) {
    case FsType::NONE:
        return "none";
    case FsType::LOCAL:
        return "local";
    case FsType::NETWORK:
        return "network";
    }
    return "unknown";
}

std::string_view RankInfo::hostname_view() const noexcept {
    return {hostname.data(), ::strnlen(hostname.data(), hostname.size())};
}

namespace {

/// Component count of @p prefix if it is a path prefix of @p path, else -1.
std::ptrdiff_t prefix_length(
    std::filesystem::path const& prefix, std::filesystem::path const& path
) {
    auto [prefix_end, _] = std::ranges::mismatch(prefix, path);
    return prefix_end == prefix.end() ? std::distance(prefix.begin(), prefix_end) : -1;
}

/// First line of a file, or "" if unreadable.
std::string read_first_line(char const* path) {
    std::ifstream file{path};
    std::string line;
    std::getline(file, line);
    return line;
}

/// Classify @p spill_dir (nullptr if no `DiskResource`) by the longest network mount
/// that contains it; `LOCAL` if none does.
SpillDirInfo classify_spill_dir(std::filesystem::path const* spill_dir) {
    if (spill_dir == nullptr) {
        return {.mount_id = 0, .fs_type = FsType::NONE};
    }
    std::error_code ec;
    auto dir = std::filesystem::weakly_canonical(*spill_dir, ec);
    if (ec) {
        dir = std::filesystem::absolute(*spill_dir).lexically_normal();
    }

    NetworkMount const* best = nullptr;
    std::ptrdiff_t best_len = -1;
    for (auto const& mount : get_network_mounts()) {
        auto const len = prefix_length(
            std::filesystem::path{mount.mount_point}.lexically_normal(), dir
        );
        if (len > best_len) {  // A mount that isn't a prefix has length -1.
            best = &mount;
            best_len = len;
        }
    }
    if (best == nullptr) {
        return {.mount_id = 0, .fs_type = FsType::LOCAL};
    }
    return {
        .mount_id = fnv1a({best->fs_type, best->source, best->mount_point}),
        .fs_type = FsType::NETWORK
    };
}

/// Collect the `RankInfo` of the calling process.
RankInfo local_rank_info(Rank rank, std::filesystem::path const* spill_dir) {
    RankInfo info{};  // Zero every byte, including padding, before filling.
    info.rank = rank;

    if (::gethostname(info.hostname.data(), info.hostname.size()) != 0) {
        info.hostname.fill('\0');
    }
    info.hostname.back() = '\0';
    info.node_id =
        fnv1a({info.hostname_view(), read_first_line("/proc/sys/kernel/random/boot_id")});

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        device = -1;
        std::ignore = cudaGetLastError();  // Do not leave the error for later calls.
    }
    info.gpu_device = static_cast<std::int32_t>(device);

    struct stat ns_stat{};
    if (::stat("/proc/self/ns/mnt", &ns_stat) == 0) {
        info.mount_ns = static_cast<std::uint64_t>(ns_stat.st_ino);
    }

    info.spill_dir = classify_spill_dir(spill_dir);
    return info;
}

/// Allgather the `RankInfo` of every rank (as raw metadata bytes), indexed by rank.
std::vector<RankInfo> allgather_rankinfo(
    std::shared_ptr<Communicator> const& comm,
    OpID op_id,
    BufferResource* br,
    RankInfo const& local
) {
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(sizeof(RankInfo));
    std::memcpy(metadata->data(), &local, sizeof(RankInfo));
    auto res = br->reserve_or_fail(0, MemoryType::HOST);
    auto data = br->make_buffer(br->stream_pool()->get_stream(), std::move(res));

    coll::AllGather allgather{comm, op_id, br};
    allgather.insert(0, PackedData{std::move(metadata), std::move(data)});
    allgather.insert_finished();
    // With one insert per rank, ordered extraction puts rank `r`'s info at index `r`.
    auto const gathered = allgather.wait_and_extract(coll::AllGather::Ordered::YES);
    RAPIDSMPF_EXPECTS(
        std::cmp_equal(gathered.size(), comm->nranks()),
        "allgather_rankinfo: expected one RankInfo per rank, got "
            + std::to_string(gathered.size()),
        std::runtime_error
    );

    std::vector<RankInfo> ret(gathered.size());
    for (std::size_t r = 0; r < gathered.size(); ++r) {
        auto const& bytes = *gathered[r].metadata;
        RAPIDSMPF_EXPECTS(
            bytes.size() == sizeof(RankInfo),
            "allgather_rankinfo: RankInfo size mismatch; all ranks must run the same "
            "build",
            std::runtime_error
        );
        std::memcpy(&ret[r], bytes.data(), sizeof(RankInfo));
    }
    return ret;
}

}  // namespace

std::shared_ptr<RankTopology> RankTopology::create(
    std::shared_ptr<Communicator> const& comm, std::shared_ptr<BufferResource> const& br
) {
    RAPIDSMPF_EXPECTS(comm != nullptr, "comm cannot be null", std::invalid_argument);
    RAPIDSMPF_EXPECTS(br != nullptr, "br cannot be null", std::invalid_argument);

    auto const& disk = br->disk_resource();
    RankInfo const local =
        local_rank_info(comm->rank(), disk != nullptr ? &disk->directory() : nullptr);

    // The constructor checks `ranks[r].rank == r`.
    return std::make_shared<RankTopology>(
        comm->rank(),
        allgather_rankinfo(comm, detail::rank_topology_op_id, br.get(), local)
    );
}

RankTopology::RankTopology(Rank rank, std::vector<RankInfo> ranks)
    : rank_{rank}, ranks_{std::move(ranks)} {
    auto const my_node = info(rank_).node_id;  // Throws if `rank` is out of range.
    RAPIDSMPF_EXPECTS(
        std::ranges::equal(
            ranks_ | std::views::transform(&RankInfo::rank),
            std::views::iota(Rank{0}, static_cast<Rank>(ranks_.size()))
        ),
        "RankTopology: ranks must be indexed by rank",
        std::invalid_argument
    );
    for (auto const& ri : ranks_) {
        if (ri.node_id == my_node) {
            if (ri.rank == rank_) {
                local_rank_idx_ = node_local_ranks_.size();
            }
            node_local_ranks_.push_back(ri.rank);
        }
    }
}

std::string RankTopology::str() const {
    std::stringstream ss;
    ss << "RankTopology(rank=" << rank_ << ", nranks=" << ranks_.size()
       << ", local_rank_idx=" << local_rank_idx_ << ", node_local_ranks=[";
    for (std::size_t i = 0; i < node_local_ranks_.size(); ++i) {
        ss << (i == 0 ? "" : ", ") << node_local_ranks_[i];
    }
    ss << "], ranks=[";
    for (std::size_t i = 0; i < ranks_.size(); ++i) {
        auto const& r = ranks_[i];
        ss << (i == 0 ? "" : ", ") << "{rank=" << r.rank << ", host=" << r.hostname_view()
           << ", gpu=" << r.gpu_device << ", spill=" << to_string(r.spill_dir.fs_type)
           << "}";
    }
    ss << "])";
    return ss.str();
}

}  // namespace rapidsmpf
