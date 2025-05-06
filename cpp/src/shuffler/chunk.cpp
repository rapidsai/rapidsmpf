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

Chunk::Chunk(
    PartID pid,
    ChunkID cid,
    std::size_t gpu_data_size,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> gpu_data
)
    : pid{pid},
      cid{cid},
      expected_num_chunks{0},
      gpu_data_size{gpu_data_size},
      metadata{std::move(metadata)},
      gpu_data{std::move(gpu_data)} {}

Chunk::Chunk(PartID pid, ChunkID cid, std::size_t expected_num_chunks)
    : pid{pid},
      cid{cid},
      expected_num_chunks{expected_num_chunks},
      gpu_data_size{0},
      metadata{nullptr},
      gpu_data{nullptr} {}

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

bool Chunk::is_ready() const {
    return (expected_num_chunks > 0) || (gpu_data_size == 0)
           || (gpu_data && gpu_data->is_ready());
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

ChunkBatch ChunkBatch::get_data(
    ChunkID new_chunk_id, size_t i, rmm::cuda_stream_view /* stream */
) {
    RAPIDSMPF_EXPECTS(i < n_messages(), "index out of bounds", std::out_of_range);

    if (is_control_message(i)) {
        return from_finished_partition(new_chunk_id, part_id(i), expected_num_chunks(i));
    }

    ChunkBatch new_chunk;
    if (n_messages() == 1) {  // i == 0, already verified
        // If there is only one message, move the metadata and data to the new chunk.
        new_chunk.metadata_ = std::move(metadata_);
        new_chunk.data_ = std::move(data_);
    } else {
        RAPIDSMPF_EXPECTS(false, "not implemented");
        // TODO: slice and copy data
    }

    return new_chunk;
}

ChunkBatch ChunkBatch::from_packed_data(
    ChunkID chunk_id,
    PartID part_id,
    PackedData&& packed_data,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    ChunkBatch chunk;
    size_t metadata_buf_size =
        metadata_message_header_size(1)
        + (packed_data.metadata ? packed_data.metadata->size() : 0);
    // Create metadata buffer
    chunk.metadata_ = std::make_unique<std::vector<uint8_t>>(metadata_buf_size);

    // Write chunk ID
    *reinterpret_cast<ChunkID*>(chunk.metadata_->data()) = chunk_id;

    // Write number of messages (1)
    *reinterpret_cast<size_t*>(chunk.metadata_->data() + sizeof(ChunkID)) = 1;

    // Write partition ID
    *reinterpret_cast<PartID*>(chunk.part_ids_begin()) = part_id;

    // Write expected number of chunks (0 for data message)
    *reinterpret_cast<size_t*>(chunk.expected_num_chunks_begin()) = 0;

    // Write metadata size
    if (packed_data.metadata) {
        RAPIDSMPF_EXPECTS(
            packed_data.metadata->size() <= std::numeric_limits<uint32_t>::max(),
            "metadata size is too large(> 4GB)"
        );
        *reinterpret_cast<uint32_t*>(chunk.psum_meta_begin()) =
            packed_data.metadata->size();

        // Copy data into the chunk's metadata buffer
        std::memcpy(
            chunk.concat_metadata_begin(),
            packed_data.metadata->data(),
            packed_data.metadata->size()
        );

        assert(packed_data.metadata->size() == chunk.concat_metadata_size());
    }

    if (packed_data.gpu_data) {
        // Write data size
        *reinterpret_cast<uint64_t*>(chunk.psum_data_begin()) =
            packed_data.gpu_data->size();
        chunk.data_ = br->move(
            std::move(packed_data.gpu_data),
            stream,
            std::make_shared<Buffer::Event>(stream)
        );
    }

    return chunk;
}

ChunkBatch ChunkBatch::from_finished_partition(
    ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
) {
    ChunkBatch chunk;
    size_t metadata_buf_size = metadata_message_header_size(1);
    // Create metadata buffer
    chunk.metadata_ = std::make_unique<std::vector<uint8_t>>(metadata_buf_size, 0);

    // Write chunk ID
    *reinterpret_cast<ChunkID*>(chunk.metadata_->data()) = chunk_id;

    // Write number of messages (1)
    *reinterpret_cast<size_t*>(chunk.metadata_->data() + sizeof(ChunkID)) = 1;

    // Write partition ID
    *reinterpret_cast<PartID*>(chunk.part_ids_begin()) = part_id;

    // Write expected number of chunks
    *reinterpret_cast<size_t*>(chunk.expected_num_chunks_begin()) = expected_num_chunks;

    return chunk;
}

ChunkBatch ChunkBatch::from_metadata_message(
    std::unique_ptr<std::vector<uint8_t>> msg, bool validate
) {
    if (validate) {
        RAPIDSMPF_EXPECTS(
            validate_metadata_format(*msg),
            "metadata buffer does not follow the expected format"
        );
    }
    ChunkBatch chunk;
    chunk.metadata_ = std::move(msg);
    return chunk;
}

bool ChunkBatch::validate_metadata_format(std::vector<uint8_t> const& metadata_buf) {
    // Check if buffer is large enough to contain at least the header
    if (metadata_buf.size() < sizeof(ChunkID) + sizeof(size_t)) {
        return false;
    }

    // Get number of messages
    size_t n_messages =
        *reinterpret_cast<size_t const*>(metadata_buf.data() + sizeof(ChunkID));

    if (n_messages == 0) {  // no messages
        return false;
    }

    // Check if buffer is large enough to contain all the messages' metadata
    size_t header_size = metadata_message_header_size(n_messages);
    if (metadata_buf.size() < header_size) {
        return false;
    }

    // For each message, validate the metadata and data sizes
    auto const* psum_meta = reinterpret_cast<uint32_t const*>(
        metadata_buf.data() + sizeof(ChunkID) + sizeof(size_t)
        + n_messages * (sizeof(PartID) + sizeof(size_t))
    );
    auto const* psum_data = reinterpret_cast<uint64_t const*>(psum_meta + n_messages);

    // Check if prefix sums are non-decreasing
    bool is_non_decreasing = true;
    for (size_t i = 1; i < n_messages; ++i) {
        is_non_decreasing &= (psum_meta[i] >= psum_meta[i - 1]);
        is_non_decreasing &= (psum_data[i] >= psum_data[i - 1]);
    }
    if (!is_non_decreasing) {
        return false;
    }

    // Check if the total metadata size matches the buffer size
    size_t total_meta_size = psum_meta[n_messages - 1];
    if (metadata_buf.size() != header_size + total_meta_size) {
        return false;
    }

    return true;
}

std::string ChunkBatch::str(std::size_t /*max_nbytes*/, rmm::cuda_stream_view /*stream*/)
    const {
    std::stringstream ss;
    ss << "Chunk(id=" << chunk_id() << ", n=" << n_messages();

    // Add message details
    for (size_t i = 0; i < n_messages(); ++i) {
        ss << "msg[" << i << "]={";
        ss << "part_id=" << part_id(i);
        ss << ", expected_num_chunks=" << expected_num_chunks(i);
        ss << ", metadata_size=" << metadata_size(i);
        ss << ", data_size=" << data_size(i);
        ss << "},";
    }
    ss << ")";
    return ss.str();
}
}  // namespace rapidsmpf::shuffler::detail
