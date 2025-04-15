/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf::shuffler::detail {


Chunk::Chunk(
    PartID pid,
    ChunkID cid,
    std::size_t expected_num_chunks,
    std::size_t gpu_data_size,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> gpu_data
)
    : pid{pid},
      cid{cid},
      expected_num_chunks{expected_num_chunks},
      gpu_data_size{gpu_data_size},
      metadata{std::move(metadata)},
      gpu_data{std::move(gpu_data)} {}

Chunk::Chunk(PartID pid, ChunkID cid, std::size_t expected_num_chunks)
    : Chunk{pid, cid, expected_num_chunks, 0, nullptr, nullptr} {}

std::unique_ptr<std::vector<uint8_t>> Chunk::to_metadata_message() const {
    size_t metadata_size = metadata ? metadata->size() : 0;
    auto msg = std::make_unique<std::vector<uint8_t>>(
        metadata_size + sizeof(MetadataMessageHeader)
    );
    std::ignore = to_metadata_message(*msg, 0);
    return msg;
}

std::ptrdiff_t Chunk::to_metadata_message(
    std::vector<uint8_t>& msg, std::ptrdiff_t offset
) const {
    size_t metadata_size = metadata ? metadata->size() : 0;
    // We need at least (sizeof(MetadataMessageHeader) + metadata_size) amount of space
    // from the offset
    RAPIDSMP_EXPECTS(
        size_t(offset) + sizeof(MetadataMessageHeader) + metadata_size <= msg.size(),
        "insufficient space in the buffer to copy metadata"
    );
    // Write the header in the first part of `msg`.
    *reinterpret_cast<MetadataMessageHeader*>(msg.data() + offset) = {
        pid, cid, expected_num_chunks, metadata_size, gpu_data_size
    };
    // Then place the metadata afterwards.
    if (metadata_size > 0) {
        std::memcpy(
            msg.data() + offset + sizeof(MetadataMessageHeader),
            metadata->data(),
            metadata_size
        );
        metadata->clear();
    }
    return std::ptrdiff_t(sizeof(MetadataMessageHeader) + metadata_size);
}

Chunk Chunk::from_metadata_message(std::unique_ptr<std::vector<uint8_t>> const& msg) {
    auto header = reinterpret_cast<MetadataMessageHeader const*>(msg->data());
    std::unique_ptr<std::vector<uint8_t>> metadata;
    if (msg->size() > sizeof(MetadataMessageHeader)) {
        metadata = std::make_unique<std::vector<uint8_t>>(
            msg->begin() + sizeof(MetadataMessageHeader), msg->end()
        );
    }
    return Chunk{
        header->pid,
        header->cid,
        header->expected_num_chunks,
        header->gpu_data_size,
        std::move(metadata),
        nullptr
    };
}

std::unique_ptr<cudf::table> Chunk::unpack(rmm::cuda_stream_view stream) const {
    RAPIDSMPF_EXPECTS(metadata && gpu_data, "both meta and gpu data must be non-null");
    auto br = gpu_data->br;

    // Since we cannot spill, we allow and ignore overbooking.
    auto [reservation, _] = br->reserve(MemoryType::DEVICE, gpu_data->size * 2, true);

    // Copy data.
    auto meta = std::make_unique<std::vector<uint8_t>>(*metadata);
    auto gpu = br->move_to_device_buffer(
        br->copy(MemoryType::DEVICE, gpu_data, stream, reservation), stream, reservation
    );

    std::vector<PackedData> packed_vec;
    packed_vec.emplace_back(std::move(meta), std::move(gpu));
    return unpack_and_concat(std::move(packed_vec), stream, br->device_mr());
}

std::string Chunk::str(std::size_t max_nbytes, rmm::cuda_stream_view stream) const {
    std::stringstream ss;
    ss << "Chunk(pid=" << pid;
    ss << ", cid=" << cid;
    ss << ", expected_num_chunks=" << expected_num_chunks;
    ss << ", gpu_data_size=" << gpu_data_size;
    if (metadata && gpu_data && gpu_data->size < max_nbytes) {
        ss << ", " << rapidsmpf::str(unpack(stream)->view());
    } else {
        ss << ", metadata=";
        if (metadata) {
            ss << "<" << metadata->size() << "B>";
        } else {
            ss << "NULL";
        }
        ss << ", gpu_data=";
        if (gpu_data) {
            ss << "<" << gpu_data->size << "B>";
        } else {
            ss << "NULL";
        }
    }
    ss << ")";
    return ss.str();
}
}  // namespace rapidsmpf::shuffler::detail
