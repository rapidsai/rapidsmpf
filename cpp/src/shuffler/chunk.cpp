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


/**
 * @brief Try to allocate a buffer from the buffer resource.
 * @param br The buffer resource.
 * @param size The size of the buffer to allocate.
 * @return A unique pointer to the allocated buffer, or nullptr if no buffer was
 * allocated.
 */
MemoryReservation try_reserve_or_fail(BufferResource* br, size_t size) {
    // try to allocate data buffer from memory types in order [DEVICE, HOST]
    for (auto mem_type : MEMORY_TYPES) {
        auto [res, overbooking] = br->reserve(mem_type, size, false);
        if (res.size() == size) {
            return std::move(res);
        }
    }
    RAPIDSMPF_FAIL("failed to reserve memory");
}

Chunk::Chunk(
    ChunkID chunk_id,
    std::vector<PartID> part_ids,
    std::vector<size_t> expected_num_chunks,
    std::vector<uint32_t> meta_offsets,
    std::vector<uint64_t> data_offsets,
    std::unique_ptr<std::vector<uint8_t>> metadata,
    std::unique_ptr<Buffer> data
)
    : chunk_id_{chunk_id},
      part_ids_{std::move(part_ids)},
      expected_num_chunks_{std::move(expected_num_chunks)},
      meta_offsets_{std::move(meta_offsets)},
      data_offsets_{std::move(data_offsets)},
      metadata_{std::move(metadata)},
      data_{std::move(data)} {
    RAPIDSMPF_EXPECTS(
        (part_ids_.size() > 0) && (part_ids_.size() == expected_num_chunks_.size())
            && (part_ids_.size() == meta_offsets_.size())
            && (part_ids_.size() == data_offsets_.size()),
        "invalid chunk: input vectors have different sizes"
    );
}

Chunk Chunk::get_data(
    ChunkID new_chunk_id, size_t i, rmm::cuda_stream_view stream, BufferResource* br
) {
    RAPIDSMPF_EXPECTS(i < n_messages(), "index out of bounds", std::out_of_range);

    if (is_control_message(i)) {
        return from_finished_partition(new_chunk_id, part_id(i), expected_num_chunks(i));
    }

    if (n_messages() == 1) {
        // If there is only one message, move the metadata and data to the new chunk.
        return Chunk(
            new_chunk_id,
            {part_ids_[0]},
            {expected_num_chunks_[0]},
            {meta_offsets_[0]},
            {data_offsets_[0]},
            std::move(metadata_),
            data_ ? std::move(data_) : br->allocate_empty_host_buffer()
        );
    } else {
        // copy the metadata to the new chunk
        uint32_t meta_slice_size = metadata_size(i);
        std::ptrdiff_t meta_slice_offset =
            (i == 0 ? 0 : std::ptrdiff_t(meta_offsets_[i - 1]));
        std::vector<uint8_t> meta_slice(meta_slice_size);
        std::memcpy(
            meta_slice.data(), metadata_->data() + meta_slice_offset, meta_slice_size
        );

        // copy the data to the new chunk
        size_t data_slice_size = data_size(i);
        std::unique_ptr<Buffer> data_slice;
        if (data_slice_size == 0) {
            data_slice = br->allocate_empty_host_buffer();
        } else {
            std::ptrdiff_t data_slice_offset =
                (i == 0 ? 0 : std::ptrdiff_t(data_offsets_[i - 1]));
            auto reserve = try_reserve_or_fail(br, data_slice_size);
            data_slice =
                data_->copy_slice(data_slice_offset, data_slice_size, reserve, stream);
        }

        return {
            new_chunk_id,
            {part_ids_[i]},
            {0},
            {meta_slice_size},
            {data_slice_size},
            std::make_unique<std::vector<uint8_t>>(std::move(meta_slice)),
            std::move(data_slice)
        };
    }
}

Chunk Chunk::from_packed_data(
    ChunkID chunk_id,
    PartID part_id,
    PackedData&& packed_data,
    std::shared_ptr<Buffer::Event> event,
    rmm::cuda_stream_view stream,
    BufferResource* br
) {
    std::vector<uint32_t> meta_offsets{0};
    RAPIDSMPF_EXPECTS(packed_data.metadata != nullptr, "packed_data.metadata is nullptr");
    meta_offsets[0] = static_cast<uint32_t>(packed_data.metadata->size());

    std::vector<uint64_t> data_offsets{0};
    if (packed_data.gpu_data) {
        data_offsets[0] = packed_data.gpu_data->size();
    }

    return {
        chunk_id,
        {part_id},
        {0},  // expected_num_chunks
        std::move(meta_offsets),
        std::move(data_offsets),
        std::move(packed_data.metadata),
        packed_data.gpu_data
            ? br->move(std::move(packed_data.gpu_data), stream, std::move(event))
            : nullptr
    };
}

Chunk Chunk::from_finished_partition(
    ChunkID chunk_id, PartID part_id, size_t expected_num_chunks
) {
    return {chunk_id, {part_id}, {expected_num_chunks}, {0}, {0}};
}

Chunk Chunk::deserialize(std::vector<uint8_t> const& msg, bool validate) {
    if (validate) {
        RAPIDSMPF_EXPECTS(
            validate_format(msg), "serialized message does not follow the expected format"
        );
    }
    size_t offset = 0;

    ChunkID chunk_id;
    std::memcpy(&chunk_id, msg.data() + offset, sizeof(ChunkID));
    offset += sizeof(ChunkID);

    size_t n_messages;
    std::memcpy(&n_messages, msg.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);

    std::vector<PartID> part_ids(n_messages);
    std::memcpy(part_ids.data(), msg.data() + offset, n_messages * sizeof(PartID));
    offset += n_messages * sizeof(PartID);

    std::vector<size_t> expected_num_chunks(n_messages);
    std::memcpy(
        expected_num_chunks.data(), msg.data() + offset, n_messages * sizeof(size_t)
    );
    offset += n_messages * sizeof(size_t);

    std::vector<uint32_t> meta_offsets(n_messages);
    std::memcpy(meta_offsets.data(), msg.data() + offset, n_messages * sizeof(uint32_t));
    offset += n_messages * sizeof(uint32_t);

    std::vector<uint64_t> data_offsets(n_messages);
    std::memcpy(data_offsets.data(), msg.data() + offset, n_messages * sizeof(uint64_t));
    offset += n_messages * sizeof(uint64_t);

    auto concat_metadata = std::make_unique<std::vector<uint8_t>>(
        msg.begin() + static_cast<int64_t>(offset), msg.end()
    );

    return {
        chunk_id,
        std::move(part_ids),
        std::move(expected_num_chunks),
        std::move(meta_offsets),
        std::move(data_offsets),
        std::move(concat_metadata),
        nullptr
    };
}

bool Chunk::validate_format(std::vector<uint8_t> const& serialized_buf) {
    // Check if buffer is large enough to contain at least the header
    if (serialized_buf.size() < sizeof(ChunkID) + sizeof(size_t)) {
        return false;
    }

    // Get number of messages
    size_t n = 0;
    std::memcpy(&n, serialized_buf.data() + sizeof(ChunkID), sizeof(size_t));

    if (n == 0) {  // no messages
        return false;
    }

    // Check if buffer is large enough to contain all the messages' metadata
    size_t header_size = metadata_message_header_size(n);
    if (serialized_buf.size() < header_size) {
        return false;
    }

    // For each message, validate the metadata and data sizes
    uint8_t const* meta_offsets_start =
        serialized_buf.data()
        + (sizeof(ChunkID) + sizeof(size_t) + n * (sizeof(PartID) + sizeof(size_t)));
    uint8_t const* data_offsets_start = meta_offsets_start + n * sizeof(uint32_t);

    // Check if prefix sums are non-decreasing
    bool is_non_decreasing = true;
    for (size_t i = 1; i < n; ++i) {
        uint32_t prev_meta_offset, this_meta_offset;
        std::memcpy(
            &prev_meta_offset,
            meta_offsets_start + (i - 1) * sizeof(uint32_t),
            sizeof(uint32_t)
        );
        std::memcpy(
            &this_meta_offset, meta_offsets_start + i * sizeof(uint32_t), sizeof(uint32_t)
        );
        is_non_decreasing &= (this_meta_offset >= prev_meta_offset);

        uint64_t prev_data_offset, this_data_offset;
        std::memcpy(
            &prev_data_offset,
            data_offsets_start + (i - 1) * sizeof(uint64_t),
            sizeof(uint64_t)
        );
        std::memcpy(
            &this_data_offset, data_offsets_start + i * sizeof(uint64_t), sizeof(uint64_t)
        );
        is_non_decreasing &= (this_data_offset >= prev_data_offset);
    }
    if (!is_non_decreasing) {
        return false;
    }

    // Check if the total metadata size matches the buffer size
    uint32_t total_meta_size;
    std::memcpy(
        &total_meta_size,
        meta_offsets_start + (n - 1) * sizeof(uint32_t),
        sizeof(uint32_t)
    );
    if (serialized_buf.size() != header_size + total_meta_size) {
        return false;
    }

    return true;
}

std::string Chunk::str(std::size_t /*max_nbytes*/, rmm::cuda_stream_view /*stream*/)
    const {
    std::stringstream ss;
    ss << "Chunk(id=" << chunk_id() << ", n=" << n_messages() << ", ";

    // Add message details
    for (size_t i = 0; i < n_messages(); ++i) {
        ss << "msg[" << i << "]={";
        ss << "part_id=" << part_ids_[i];
        ss << ", expected_num_chunks=" << expected_num_chunks_[i];
        ss << ", metadata_size=" << (meta_offsets_.empty() ? 0 : meta_offsets_[i]);
        ss << ", data_size=" << (data_offsets_.empty() ? 0 : data_offsets_[i]);
        ss << "},";
    }
    ss << ")";
    return ss.str();
}

std::unique_ptr<std::vector<uint8_t>> Chunk::serialize() const {
    size_t n = this->n_messages();

    size_t metadata_buf_size =
        metadata_message_header_size(n) + (metadata_ ? metadata_->size() : 0);
    auto metadata_buf = std::make_unique<std::vector<uint8_t>>(metadata_buf_size);

    uint8_t* p = metadata_buf->data();
    // Write chunk ID
    std::memcpy(p, &chunk_id_, sizeof(ChunkID));
    p += sizeof(ChunkID);

    // Write number of messages
    std::memcpy(p, &n, sizeof(size_t));
    p += sizeof(size_t);

    // Write partition IDs
    std::memcpy(p, part_ids_.data(), n * sizeof(PartID));
    p += n * sizeof(PartID);

    // Write expected number of chunks
    std::memcpy(p, expected_num_chunks_.data(), n * sizeof(size_t));
    p += n * sizeof(size_t);

    // Write metadata offsets
    std::memcpy(p, meta_offsets_.data(), n * sizeof(uint32_t));
    p += n * sizeof(uint32_t);

    // Write data offsets
    std::memcpy(p, data_offsets_.data(), n * sizeof(uint64_t));
    p += n * sizeof(uint64_t);

    // Write concatenated metadata
    if (metadata_) {
        std::memcpy(p, metadata_->data(), metadata_->size());
        metadata_->clear();
    }

    return metadata_buf;
}

ChunkBuilder::ChunkBuilder(
    rmm::cuda_stream_view stream, BufferResource* br, size_t num_messages_hint
)
    : stream_(stream), br_(br) {
    if (num_messages_hint > 0) {
        part_ids_.reserve(num_messages_hint);
        expected_num_chunks_.reserve(num_messages_hint);
        meta_offsets_.reserve(num_messages_hint);
        data_offsets_.reserve(num_messages_hint);
        staged_metadata_.reserve(num_messages_hint);
    }
}

ChunkBuilder& ChunkBuilder::add_control_message(
    PartID part_id, size_t expected_num_chunks
) {
    part_ids_.push_back(part_id);
    expected_num_chunks_.push_back(expected_num_chunks);
    // For control messages, we need to add zero offsets since they don't have data
    if (meta_offsets_.empty() && data_offsets_.empty()) {
        meta_offsets_.push_back(0);
        data_offsets_.push_back(0);
    } else {
        meta_offsets_.push_back(meta_offsets_.back());
        data_offsets_.push_back(data_offsets_.back());
    }
    return *this;
}

ChunkBuilder& ChunkBuilder::add_packed_data(PartID part_id, PackedData&& packed_data) {
    // Calculate metadata offset
    RAPIDSMPF_EXPECTS(packed_data.metadata != nullptr, "packed_data.metadata is nullptr");
    uint32_t meta_offset = meta_offsets_.empty() ? 0 : meta_offsets_.back();
    meta_offset += packed_data.metadata->size();
    meta_offsets_.push_back(meta_offset);

    // Calculate data offset
    uint64_t data_offset = data_offsets_.empty() ? 0 : data_offsets_.back();
    if (packed_data.gpu_data) {
        data_offset += packed_data.gpu_data->size();
    }
    data_offsets_.push_back(data_offset);

    // Add the part id and expected number of chunks
    part_ids_.push_back(part_id);
    expected_num_chunks_.push_back(0);  // Data messages have 0 expected chunks

    // Move the packed data into our staged buffers
    if (packed_data.metadata) {
        staged_metadata_.emplace_back(std::move(*packed_data.metadata));
    }

    if (packed_data.gpu_data) {
        // trivially convert the rmm buffer to a Buffer.
        // TODO: based on the current Buffer API, we are needlessly create an event for
        // this operation. rmm buffer is only staged, until the chunk is built. During the
        // build() method, a new event is created which can guarantee the completion of
        // all staged operations.
        staged_data_.emplace_back(
            br_->move(std::move(packed_data.gpu_data), stream_, nullptr)
        );
    }

    return *this;
}

Chunk ChunkBuilder::build(ChunkID chunk_id) {
    RAPIDSMPF_EXPECTS(
        !part_ids_.empty(), "No messages added to the chunk builder", std::runtime_error
    );

    // Concatenate metadata
    auto metadata = std::make_unique<std::vector<uint8_t>>(meta_offsets_.back());
    size_t meta_offset = 0;
    for (auto&& meta : staged_metadata_) {
        auto temp = std::move(meta);  // Move the vector to temp and destroy it
        std::memcpy(metadata->data() + meta_offset, temp.data(), temp.size());
        meta_offset += temp.size();
    }

    // Concatenate data
    std::unique_ptr<Buffer> data;
    size_t total_data_size = data_offsets_.back();
    if (total_data_size > 0) {
        assert(!staged_data_.empty());
        // TODO(niranda): Handle spiiling

        // try to allocate data buffer from memory types in order [DEVICE, HOST]
        auto reserve = try_reserve_or_fail(br_, total_data_size);
        data = br_->allocate(reserve.mem_type(), total_data_size, stream_, reserve);
        RAPIDSMPF_EXPECTS(reserve.size() == 0, "didn't use all of the reservation");

        // Copy the data from the staged buffers to the data buffer
        std::ptrdiff_t data_offset = 0;
        // if the data buffer is on the device, we need to create an event to track the
        // async copies
        bool need_event = (data->mem_type() == MemoryType::DEVICE);

        for (auto&& staged_buf : staged_data_) {
            auto temp = std::move(staged_buf);
            // if staged buffer is empty, skip it
            if (temp == nullptr || temp->size == 0) {
                continue;
            }

            // copy the staged buffer to the data buffer
            data_offset += temp->copy_to(*data, data_offset, stream_);

            // if the staged buffer is on the device, we need an event
            need_event |= (temp->mem_type() == MemoryType::DEVICE);
        }
        RAPIDSMPF_EXPECTS(size_t(data_offset) == total_data_size, "didn't copy all data");

        if (need_event) {  // create a new event to track the async copies
            data->override_event(std::make_shared<Buffer::Event>(stream_));
        }
    } else {  // No data messages, so let's allocate an empty host buffer
        auto [res, size] = br_->reserve(MemoryType::HOST, 0, false);
        data = br_->allocate(MemoryType::HOST, 0, stream_, res);
    }

    return {
        chunk_id,
        std::move(part_ids_),
        std::move(expected_num_chunks_),
        std::move(meta_offsets_),
        std::move(data_offsets_),
        std::move(metadata),
        std::move(data)
    };
}

}  // namespace rapidsmpf::shuffler::detail
