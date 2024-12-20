/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>
#include <rapidsmp/error.hpp>
#include <rapidsmp/shuffler/chunk.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp::shuffler::detail {


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
    auto metadata_size = metadata ? metadata->size() : 0;
    auto msg = std::make_unique<std::vector<uint8_t>>(
        metadata_size + sizeof(MetadataMessageHeader)
    );
    // Write the header in the first part of `msg`.
    *reinterpret_cast<MetadataMessageHeader*>(msg->data()
    ) = {pid, cid, expected_num_chunks, gpu_data_size};
    // Then place the metadata afterwards.
    if (metadata_size > 0) {
        std::copy(
            metadata->begin(),
            metadata->end(),
            msg->begin() + sizeof(MetadataMessageHeader)
        );
        metadata->clear();
    }
    return msg;
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
    RAPIDSMP_EXPECTS(metadata && gpu_data, "both meta and gpu data must be non-null");
    auto br = gpu_data->br;
    auto [reservation, overbooking] = br->reserve(MemoryType::DEVICE, gpu_data->size * 2);
    // TODO: check overbooking, do we need to spill?

    // Copy data.
    auto meta = std::make_unique<std::vector<uint8_t>>(*metadata);
    auto gpu = br->move_to_device_buffer(
        br->copy(MemoryType::DEVICE, gpu_data, stream, reservation), stream, reservation
    );

    std::vector<cudf::packed_columns> packed_vec;
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
        ss << ", " << rapidsmp::str(unpack(stream)->view());
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
}  // namespace rapidsmp::shuffler::detail
