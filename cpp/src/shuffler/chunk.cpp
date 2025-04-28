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

Chunk::Event::Event(rmm::cuda_stream_view stream, Communicator::Logger& log) : log_(log) {
    RAPIDSMPF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    RAPIDSMPF_CUDA_TRY(cudaEventRecord(event_, stream));
}

Chunk::Event::~Event() {
    if (!is_done()) {
        log_.warn("Event destroyed before CUDA event completed");
    }
    cudaEventDestroy(event_);
}

[[nodiscard]] bool Chunk::Event::is_done() {
    if (!done_) {
        done_ = cudaEventQuery(event_) == cudaSuccess;
    }
    return done_;
}

Chunk::Chunk(
    PartID pid,
    ChunkID cid,
    std::size_t expected_num_chunks,
    std::size_t gpu_data_size,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> gpu_data,
    std::shared_ptr<Event> event
)
    : pid{pid},
      cid{cid},
      expected_num_chunks{expected_num_chunks},
      gpu_data_size{gpu_data_size},
      metadata{std::move(metadata)},
      gpu_data{std::move(gpu_data)},
      event{std::move(event)} {}

Chunk::Chunk(PartID pid, ChunkID cid, std::size_t expected_num_chunks)
    : Chunk{pid, cid, expected_num_chunks, 0, nullptr, nullptr, nullptr} {}

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
        nullptr,
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
