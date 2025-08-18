/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/integrations/cudf/utils.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>>
partition_and_split(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    bool allow_overbooking
) {
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    if (table.num_rows() == 0) {
        // Return views of a copy of the empty `table`.
        auto owner = std::make_unique<cudf::table>(table, stream, br->device_mr());
        return {
            std::vector<cudf::table_view>(
                static_cast<std::size_t>(num_partitions), owner->view()
            ),
            std::move(owner)
        };
    }

    // hash_partition does a deep-copy. Therefore, we need to reserve memory for
    // at least the size of the table.
    auto res = reserve_device_memory_and_spill(
        br, estimated_memory_usage(table, stream), allow_overbooking
    );
    auto [partition_table, offsets] = with_memory_reservation(std::move(res), [&] {
        return cudf::hash_partition(
            table,
            columns_to_hash,
            num_partitions,
            hash_function,
            seed,
            stream,
            br->device_mr()
        );
    });

    // Notice, the offset argument for split() and hash_partition() doesn't align.
    // hash_partition() returns the start offset of each partition thus we have to
    // skip the first offset. See: <https://github.com/rapidsai/cudf/issues/4607>.
    auto partition_offsets =
        cudf::host_span<cudf::size_type const>(offsets.data() + 1, offsets.size() - 1);

    // split does not make any copies.
    auto tbl_partitioned =
        cudf::split(partition_table->view(), partition_offsets, stream);

    return {std::move(tbl_partitioned), std::move(partition_table)};
}

std::unordered_map<shuffler::PartID, PackedData> partition_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& columns_to_hash,
    int num_partitions,
    cudf::hash_id hash_function,
    uint32_t seed,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    bool allow_overbooking
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    RAPIDSMPF_EXPECTS(num_partitions > 0, "Need to split to at least one partition");
    if (table.num_rows() == 0) {
        auto splits = std::vector<cudf::size_type>(
            static_cast<std::uint64_t>(num_partitions - 1), 0
        );
        return split_and_pack(table, splits, stream, br, statistics, allow_overbooking);
    }

    // hash_partition does a deep-copy. Therefore, we need to reserve memory for
    // at least the size of the table.
    auto res = reserve_device_memory_and_spill(
        br, estimated_memory_usage(table, stream), allow_overbooking
    );
    auto [reordered, split_points] = with_memory_reservation(std::move(res), [&] {
        return cudf::hash_partition(
            table,
            columns_to_hash,
            num_partitions,
            hash_function,
            seed,
            stream,
            br->device_mr()
        );
    });
    std::vector<cudf::size_type> splits(split_points.begin() + 1, split_points.end());
    return split_and_pack(
        reordered->view(), splits, stream, br, statistics, allow_overbooking
    );
}

std::unordered_map<shuffler::PartID, PackedData> split_and_pack(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& splits,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    bool allow_overbooking
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    std::unordered_map<shuffler::PartID, PackedData> ret;

    // contiguous split does a deep-copy. Therefore, we need to reserve memory for
    // at least the size of the table.
    auto res = reserve_device_memory_and_spill(
        br, estimated_memory_usage(table, stream), allow_overbooking
    );
    auto packed = with_memory_reservation(std::move(res), [&] {
        return cudf::contiguous_split(table, splits, stream, br->device_mr());
    });

    for (shuffler::PartID i = 0; static_cast<std::size_t>(i) < packed.size(); i++) {
        auto pack = std::move(packed[i].data);
        ret.emplace(
            i,
            PackedData(
                std::move(pack.metadata), br->move(std::move(pack.gpu_data), stream)
            )
        );
    }
    return ret;
}

std::unique_ptr<cudf::table> unpack_and_concat(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics,
    bool allow_overbooking
) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    RAPIDSMPF_MEMORY_PROFILE(statistics);
    std::vector<cudf::table_view> unpacked;
    std::vector<cudf::packed_columns> references;
    unpacked.reserve(partitions.size());
    references.reserve(partitions.size());
    size_t concat_size = 0;
    for (auto& packed_data : partitions) {
        if (!packed_data.empty()) {
            concat_size += packed_data.data->size;
            unpacked.push_back(
                cudf::unpack(references.emplace_back(
                    std::move(packed_data.metadata),
                    br->move_to_device_buffer(std::move(packed_data.data))
                ))
            );
        }
    }

    // reserve memory for concatenation
    auto reservation =
        reserve_device_memory_and_spill(br, concat_size, allow_overbooking);

    return cudf::concatenate(unpacked, stream, br->device_mr());
}

std::vector<PackedData> spill_partitions(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
) {
    auto const elapsed = Clock::now();
    // Sum the total size of all packed data in device memory.
    std::size_t device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() == MemoryType::DEVICE) {
            device_size += data->size;
        }
    }
    auto [reservation, _] = br->reserve(MemoryType::HOST, device_size, false);
    // Spill each partition to host memory.
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(
            std::move(metadata), br->move(std::move(data), stream, reservation)
        );
    }
    statistics->add_duration_stat("spill-time-device-to-host", Clock::now() - elapsed);
    statistics->add_bytes_stat("spill-bytes-device-to-host", device_size);
    return ret;
}

std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    rmm::cuda_stream_view stream,
    BufferResource* br,
    bool allow_overbooking,
    std::shared_ptr<Statistics> statistics
) {
    auto const elapsed = Clock::now();
    // Sum the total size of all packed data not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() != MemoryType::DEVICE) {
            non_device_size += data->size;
        }
    }

    // This total sum is what we need to reserve before moving data to device.
    auto reservation =
        reserve_device_memory_and_spill(br, non_device_size, allow_overbooking);

    // Unspill each partition.
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(
            std::move(metadata), br->move(std::move(data), stream, reservation)
        );
    }

    statistics->add_duration_stat("spill-time-host-to-device", Clock::now() - elapsed);
    statistics->add_bytes_stat("spill-bytes-host-to-device", non_device_size);
    return ret;
}
}  // namespace rapidsmpf
