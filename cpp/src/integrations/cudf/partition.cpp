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
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/integrations/cudf/utils.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
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
    auto reservation = br->reserve_device_memory_and_spill(
        estimated_memory_usage(table, stream), allow_overbooking
    );
    auto [partition_table, offsets] = cudf::hash_partition(
        table,
        columns_to_hash,
        num_partitions,
        hash_function,
        seed,
        stream,
        br->device_mr()
    );
    reservation.clear();

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
    auto reservation = br->reserve_device_memory_and_spill(
        estimated_memory_usage(table, stream), allow_overbooking
    );
    auto [reordered, split_points] = cudf::hash_partition(
        table,
        columns_to_hash,
        num_partitions,
        hash_function,
        seed,
        stream,
        br->device_mr()
    );
    reservation.clear();
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
    auto reservation = br->reserve_device_memory_and_spill(
        estimated_memory_usage(table, stream), allow_overbooking
    );
    auto packed = cudf::contiguous_split(table, splits, stream, br->device_mr());
    reservation.clear();

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

    // Let's find the total size of the partitions and how much of the packed data we
    // need to move to device memory (unspill).
    size_t total_size = 0;
    size_t non_device_size = 0;
    for (auto& packed_data : partitions) {
        if (!packed_data.empty()) {
            total_size += packed_data.data->size;
            if (packed_data.data->mem_type() != MemoryType::DEVICE) {
                non_device_size += 0;
            }
        }
    }

    std::vector<cudf::table_view> unpacked;
    std::vector<cudf::packed_columns> references;
    std::vector<rmm::cuda_stream_view> packed_data_streams;
    unpacked.reserve(partitions.size());
    references.reserve(partitions.size());
    packed_data_streams.reserve(partitions.size());

    // Reserve device memory for the unspill AND the cudf::unpack() calls.
    auto reservation = br->reserve_device_memory_and_spill(
        total_size + non_device_size, allow_overbooking
    );
    for (auto& packed_data : partitions) {
        if (!packed_data.empty()) {
            if (packed_data.data->size > 0) {  // No need to sync empty buffers.
                packed_data_streams.push_back(packed_data.data->stream());
            }
            unpacked.push_back(
                cudf::unpack(references.emplace_back(
                    std::move(packed_data.metadata),
                    br->move_to_device_buffer(std::move(packed_data.data), reservation)
                ))
            );
        }
    }
    reservation.clear();

    // We need to synchronize `stream` with the packed_data and update their
    // underlying device buffers to use `stream` going forward. This ensures
    // the packed data are not deallocated before we have a chance to
    // concatenate them on `stream`.
    cuda_stream_join(std::views::single(stream), packed_data_streams);
    for (cudf::packed_columns& packed_columns : references) {
        packed_columns.gpu_data->set_stream(stream);
    }

    reservation = br->reserve_device_memory_and_spill(total_size, allow_overbooking);
    return cudf::concatenate(unpacked, stream, br->device_mr());
}

std::vector<PackedData> spill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    std::shared_ptr<Statistics> statistics
) {
    auto const start_time = Clock::now();
    // Sum the total size of all packed data in device memory.
    std::size_t device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() == MemoryType::DEVICE) {
            device_size += data->size;
        }
    }
    // Spill each partition to host memory.
    auto reservation = br->reserve_or_fail(device_size, SPILL_TARGET_MEMORY_TYPES);
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(std::move(metadata), br->move(std::move(data), reservation));
    }
    statistics->add_duration_stat("spill-time-device-to-host", Clock::now() - start_time);
    statistics->add_bytes_stat("spill-bytes-device-to-host", device_size);
    return ret;
}

std::vector<PackedData> unspill_partitions(
    std::vector<PackedData>&& partitions,
    BufferResource* br,
    bool allow_overbooking,
    std::shared_ptr<Statistics> statistics
) {
    auto const start_time = Clock::now();
    // Sum the total size of all packed data not in device memory already.
    std::size_t non_device_size{0};
    for (auto& [_, data] : partitions) {
        if (data->mem_type() != MemoryType::DEVICE) {
            non_device_size += data->size;
        }
    }

    // Unspill each partition.
    auto reservation =
        br->reserve_device_memory_and_spill(non_device_size, allow_overbooking);
    std::vector<PackedData> ret;
    ret.reserve(partitions.size());
    for (auto& [metadata, data] : partitions) {
        ret.emplace_back(std::move(metadata), br->move(std::move(data), reservation));
    }

    statistics->add_duration_stat("spill-time-host-to-device", Clock::now() - start_time);
    statistics->add_bytes_stat("spill-bytes-host-to-device", non_device_size);
    return ret;
}

PackedData chunked_pack(
    cudf::table_view const& table, Buffer& bounce_buf, MemoryReservation& data_res
) {
    RAPIDSMPF_EXPECTS(
        is_device_accessible(bounce_buf.mem_type()),
        "bounce buffer is not device accessible",
        std::invalid_argument
    );

    // all copies will be done on the bounce buffer's stream
    auto stream = bounce_buf.stream();
    auto* br = data_res.br();
    size_t chunk_size = bounce_buf.size;

    cudf::chunked_pack packer(table, chunk_size, stream, br->device_mr());
    auto const packed_size = packer.get_total_contiguous_size();

    // if the packed size > data reservation, and it is within the wiggle room, pad the
    // data reservation to the packed size from the same memory type.
    if (packed_size > data_res.size()) {
        if (packed_size <= data_res.size() + total_packing_wiggle_room(table)) {
            data_res =
                data_res.br()->reserve(data_res.mem_type(), packed_size, true).first;
        }
    }

    auto data_buf = br->allocate(packed_size, stream, data_res);

    bounce_buf.write_access([&](std::byte* bounce_buf_ptr, rmm::cuda_stream_view) {
        // all copies are done on the same stream, so we can omit the stream parameter
        cudf::device_span<uint8_t> buf_span(
            reinterpret_cast<uint8_t*>(bounce_buf_ptr), chunk_size
        );

        data_buf->write_access([&](std::byte* data_ptr, rmm::cuda_stream_view) {
            size_t offset = 0;
            while (packer.has_next()) {
                size_t n_bytes = packer.next(buf_span);
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    data_ptr + offset, buf_span.data(), n_bytes, cudaMemcpyDefault, stream
                ));
                offset += n_bytes;
            }
        });
    });

    return {packer.build_metadata(), std::move(data_buf)};
}

std::unique_ptr<PackedData> pack_to_host(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& host_data_res,
    float chunked_pack_buffer_size_factor,
    std::span<MemoryType const> cpack_buf_mem_types
) {
    RAPIDSMPF_EXPECTS(
        is_host_accessible(host_data_res.mem_type()),
        "memory reservation is not host accessible",
        std::invalid_argument
    );

    auto* br = host_data_res.br();

    size_t est_table_size = estimated_memory_usage(table, stream);
    {
        // make a device reservation for packing
        auto [pack_res, overbooking] =
            br->reserve(MemoryType::DEVICE, est_table_size, true);

        if (overbooking == 0) {
            // if there is enough memory to pack the table, use `cudf::pack`
            auto packed_columns = cudf::pack(table, stream, br->device_mr());
            // clear the reservation as we are done with it.
            pack_res.clear();

            // note that this is a device buffer, so we need to move it to host memory
            auto packed_data = std::make_unique<PackedData>(
                std::move(packed_columns.metadata),
                br->move(std::move(packed_columns.gpu_data), stream)
            );

            // Handle the case where `cudf::pack` allocates slightly more than
            // the input size. This can occur because cudf uses aligned
            // allocations, which may exceed the requested size. To
            // accommodate this, we allow some wiggle room.
            if (packed_data->data->size > host_data_res.size()) {
                if (packed_data->data->size
                    <= host_data_res.size() + total_packing_wiggle_room(table))
                {
                    host_data_res =
                        br->reserve(
                              host_data_res.mem_type(), packed_data->data->size, true
                        )
                            .first;
                }
            }

            // finally copy the packed data device buffer to HOST memory
            packed_data->data = br->move(std::move(packed_data->data), host_data_res);
            return packed_data;
        }
    }

    // there is not enough memory to use cudf::pack. Use chunked_pack.
    auto chunk_size = std::max(
        static_cast<size_t>(est_table_size * chunked_pack_buffer_size_factor),
        cudf_chunked_pack_min_buffer_size
    );
    auto bounce_res = br->reserve_or_fail(chunk_size, cpack_buf_mem_types);
    auto bounce_buf = br->allocate(chunk_size, stream, bounce_res);

    return std::make_unique<PackedData>(chunked_pack(table, *bounce_buf, host_data_res));
}

}  // namespace rapidsmpf
