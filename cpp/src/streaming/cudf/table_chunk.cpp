/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {
TableChunk::TableChunk(
    std::uint64_t sequence_number,
    std::unique_ptr<cudf::table> table,
    rmm::cuda_stream_view stream
)
    : sequence_number_{sequence_number}, table_{std::move(table)}, stream_{stream} {
    RAPIDSMPF_EXPECTS(
        table_ != nullptr, "table pointer cannot be null", std::invalid_argument
    );
    table_view_ = table_->view();
    data_alloc_size_[static_cast<std::size_t>(MemoryType::DEVICE)] = table_->alloc_size();
    make_available_cost_ = 0;
}

TableChunk::TableChunk(
    std::uint64_t sequence_number,
    std::unique_ptr<cudf::packed_columns> packed_columns,
    rmm::cuda_stream_view stream
)
    : sequence_number_{sequence_number},
      packed_columns_{std::move(packed_columns)},
      stream_{stream} {
    RAPIDSMPF_EXPECTS(
        packed_columns_ != nullptr,
        "packed columns pointer cannot be null",
        std::invalid_argument
    );
    table_view_ = cudf::unpack(*packed_columns_);
    data_alloc_size_[static_cast<std::size_t>(MemoryType::DEVICE)] =
        packed_columns_->gpu_data->size();
    make_available_cost_ = 0;
}

TableChunk::TableChunk(
    std::uint64_t sequence_number,
    std::unique_ptr<PackedData> packed_data,
    rmm::cuda_stream_view stream
)
    : sequence_number_{sequence_number},
      packed_data_{std::move(packed_data)},
      stream_{stream} {
    RAPIDSMPF_EXPECTS(
        packed_data_ != nullptr,
        "packed data pointer cannot be null",
        std::invalid_argument
    );
    RAPIDSMPF_EXPECTS(
        !packed_data_->empty(), "packed data cannot be empty", std::invalid_argument
    );
    data_alloc_size_[static_cast<std::size_t>(packed_data_->data->mem_type())] =
        packed_data_->data->size;
    if (packed_data_->data->mem_type() == MemoryType::HOST) {
        make_available_cost_ = packed_data_->data->size;
    } else {
        make_available_cost_ = 0;
    }
}

std::uint64_t TableChunk::sequence_number() const {
    return sequence_number_;
}

std::size_t TableChunk::data_alloc_size(MemoryType mem_type) const {
    return data_alloc_size_.at(static_cast<std::size_t>(mem_type));
}

bool TableChunk::is_available() const {
    return table_ != nullptr || packed_columns_ != nullptr;
}

std::size_t TableChunk::make_available_cost() const {
    return make_available_cost_;
}

TableChunk TableChunk::make_available(
    MemoryReservation& reservation, rmm::cuda_stream_view stream, BufferResource* br
) {
    if (is_available()) {
        return std::move(*this);
    }
    RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "packed data pointer cannot be null");
    PackedData packed_data = std::move(*packed_data_);
    return TableChunk{
        sequence_number_,
        std::make_unique<cudf::packed_columns>(
            std::move(packed_data.metadata),
            br->move_to_device_buffer(std::move(packed_data.data), stream, reservation)
        ),
        stream
    };
}

cudf::table_view TableChunk::table_view() const {
    RAPIDSMPF_EXPECTS(
        is_available(),
        "the table view is unavailable, please make sure it is "
        "unspilled and unpacked (see `make_available`).",
        std::invalid_argument
    );
    return table_view_;
}

TableChunk TableChunk::spill_to_host(rmm::cuda_stream_view stream, BufferResource* br) {
    std::unique_ptr<PackedData> packed_data = std::move(packed_data_);

    // If it isn't already, convert `table_` or `packed_columns_` to a `PackedData`.
    if (packed_data == nullptr) {
        if (table_ != nullptr) {
            // TODO: use `cudf::chunked_pack()`.
            auto packed_columns = cudf::pack(table_->view(), stream, br->device_mr());
            packed_data = std::make_unique<PackedData>(
                std::move(packed_columns.metadata),
                br->move(std::move(packed_columns.gpu_data), stream)
            );
        } else if (packed_columns_ != nullptr) {
            packed_data = std::make_unique<PackedData>(
                std::move(packed_columns_->metadata),
                br->move(std::move(packed_columns_->gpu_data), stream)
            );
        } else {
            RAPIDSMPF_FAIL("all three data pointers are null");
        }
    }
    // Spill data to host memory.
    auto [res, _] = br->reserve(MemoryType::HOST, packed_data->data->size, false);
    packed_data->data = br->move(std::move(packed_data->data), stream, res);

    return TableChunk{sequence_number_, std::move(packed_data), stream};
}

}  // namespace rapidsmpf::streaming
