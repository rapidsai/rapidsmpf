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
    : sequence_number_{sequence_number},
      table_{std::move(table)},
      stream_{stream},
      is_spillable_{true} {
    RAPIDSMPF_EXPECTS(
        table_ != nullptr, "table pointer cannot be null", std::invalid_argument
    );
    table_view_ = table_->view();
    data_alloc_size_[static_cast<std::size_t>(MemoryType::DEVICE)] = table_->alloc_size();
    make_available_cost_ = 0;
}

TableChunk::TableChunk(
    std::uint64_t sequence_number,
    cudf::table_view table_view,
    std::size_t device_alloc_size,
    rmm::cuda_stream_view stream,
    OwningWrapper&& owner,
    bool is_exclusive_view
)
    : owner_{std::move(owner)},
      sequence_number_{sequence_number},
      table_view_{table_view},
      stream_{stream},
      is_spillable_{is_exclusive_view} {
    data_alloc_size_[static_cast<std::size_t>(MemoryType::DEVICE)] = device_alloc_size;
    make_available_cost_ = 0;
}

TableChunk::TableChunk(
    std::uint64_t sequence_number,
    std::unique_ptr<cudf::packed_columns> packed_columns,
    rmm::cuda_stream_view stream
)
    : sequence_number_{sequence_number},
      packed_columns_{std::move(packed_columns)},
      stream_{stream},
      is_spillable_{true} {
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
    std::uint64_t sequence_number, std::unique_ptr<PackedData> packed_data
)
    : sequence_number_{sequence_number},
      packed_data_{std::move(packed_data)},
      stream_{packed_data_->data->stream()},
      is_spillable_{true} {
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

std::uint64_t TableChunk::sequence_number() const noexcept {
    return sequence_number_;
}

std::size_t TableChunk::data_alloc_size(MemoryType mem_type) const {
    return data_alloc_size_.at(static_cast<std::size_t>(mem_type));
}

bool TableChunk::is_available() const noexcept {
    return table_view_.has_value();
}

std::size_t TableChunk::make_available_cost() const noexcept {
    return make_available_cost_;
}

TableChunk TableChunk::make_available(MemoryReservation& reservation) {
    if (is_available()) {
        return std::move(*this);
    }
    RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "packed data pointer cannot be null");
    PackedData packed_data = std::move(*packed_data_);
    auto stream = packed_data.data->stream();
    return TableChunk{
        sequence_number_,
        std::make_unique<cudf::packed_columns>(
            std::move(packed_data.metadata),
            reservation.br()->move_to_device_buffer(
                std::move(packed_data.data), reservation
            )
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
    return table_view_.value();
}

bool TableChunk::is_spillable() const {
    return is_spillable_;
}

TableChunk TableChunk::spill_to_host(BufferResource* br) {
    RAPIDSMPF_EXPECTS(
        is_spillable_, "table chunk isn't spillable", std::invalid_argument
    );
    std::unique_ptr<PackedData> packed_data = std::move(packed_data_);

    // If it isn't already packed data, convert it.
    if (packed_data == nullptr) {
        if (table_ != nullptr) {
            // TODO: use `cudf::chunked_pack()`.
            auto packed_columns = cudf::pack(table_->view(), stream_, br->device_mr());
            packed_data = std::make_unique<PackedData>(
                std::move(packed_columns.metadata),
                br->move(std::move(packed_columns.gpu_data), stream_)
            );
        } else if (packed_columns_ != nullptr) {
            packed_data = std::make_unique<PackedData>(
                std::move(packed_columns_->metadata),
                br->move(std::move(packed_columns_->gpu_data), stream_)
            );
        } else {
            auto packed_columns = cudf::pack(table_view(), stream_, br->device_mr());
            packed_data = std::make_unique<PackedData>(
                std::move(packed_columns.metadata),
                br->move(std::move(packed_columns.gpu_data), stream_)
            );
        }
    }

    // Spill data to host memory.
    auto [res, _] = br->reserve(MemoryType::HOST, packed_data->data->size, false);
    packed_data->data = br->move(std::move(packed_data->data), res);

    return TableChunk{sequence_number_, std::move(packed_data)};
}
}  // namespace rapidsmpf::streaming
