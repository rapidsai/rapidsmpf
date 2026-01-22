/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include <rapidsmpf/integrations/cudf/utils.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {

TableChunk::TableChunk(std::unique_ptr<cudf::table> table, rmm::cuda_stream_view stream)
    : table_{std::move(table)}, stream_{stream}, is_spillable_{true} {
    RAPIDSMPF_EXPECTS(
        table_ != nullptr, "table pointer cannot be null", std::invalid_argument
    );
    table_view_ = table_->view();
    data_alloc_size_[static_cast<std::size_t>(MemoryType::DEVICE)] = table_->alloc_size();
    make_available_cost_ = 0;
}

TableChunk::TableChunk(
    cudf::table_view table_view,
    rmm::cuda_stream_view stream,
    OwningWrapper&& owner,
    ExclusiveView exclusive_view
)
    : owner_{std::move(owner)},
      table_view_{table_view},
      stream_{stream},
      is_spillable_{static_cast<bool>(exclusive_view)} {
    data_alloc_size_[static_cast<std::size_t>(MemoryType::DEVICE)] =
        estimated_memory_usage(table_view, stream_);
    make_available_cost_ = 0;
}

TableChunk::TableChunk(
    std::unique_ptr<cudf::packed_columns> packed_columns, rmm::cuda_stream_view stream
)
    : packed_columns_{std::move(packed_columns)}, stream_{stream}, is_spillable_{true} {
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

TableChunk::TableChunk(std::unique_ptr<PackedData> packed_data)
    : packed_data_{std::move(packed_data)},
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
    if (packed_data_->data->mem_type() != MemoryType::DEVICE) {
        make_available_cost_ = packed_data_->data->size;
    } else {
        make_available_cost_ = 0;
    }
}

rmm::cuda_stream_view TableChunk::stream() const noexcept {
    return stream_;
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

TableChunk TableChunk::copy(MemoryReservation& reservation) const {
    BufferResource* br = reservation.br();
    if (is_available()) {  // If `is_available() == true`, the chunk is in device memory.
        switch (reservation.mem_type()) {
        case MemoryType::DEVICE:
            {
                // Use libcudf to copy the table_view().
                auto table = std::make_unique<cudf::table>(
                    table_view(), stream(), br->device_mr()
                );
                // And update the provided `reservation`.
                br->release(reservation, data_alloc_size(MemoryType::DEVICE));
                return TableChunk(std::move(table), stream());
            }
        case MemoryType::HOST:
        case MemoryType::PINNED_HOST:
            {
                // Get the packed data either from `packed_columns_` or `table_view().
                std::unique_ptr<PackedData> packed_data;
                if (packed_columns_ != nullptr) {
                    // If `packed_columns_` is available, we copy its gpu data to a
                    // new host buffer and its metadata to a new std::vector.

                    // Copy packed_columns' metadata.
                    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
                        *packed_columns_->metadata
                    );

                    // Copy packed columns' gpu data.
                    auto gpu_data = br->allocate(
                        packed_columns_->gpu_data->size(), stream(), reservation
                    );
                    gpu_data->write_access([&](std::byte* dst,
                                               rmm::cuda_stream_view& stream) {
                        RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                            dst,
                            packed_columns_->gpu_data->data(),
                            packed_columns_->gpu_data->size(),
                            cudaMemcpyDefault,
                            stream
                        ));
                    });
                    packed_data = std::make_unique<PackedData>(
                        std::move(metadata), std::move(gpu_data)
                    );
                } else {
                    // If `packed_columns_` is not available, we use libcudf's pack() to
                    // serialize `table_view()` into a packed_columns and then we move
                    // the packed_columns' gpu_data to a new host buffer.

                    // TODO: use `cudf::chunked_pack()` with a bounce buffer. Currently,
                    // `cudf::pack()` allocates device memory we haven't reserved.
                    auto packed_columns =
                        cudf::pack(table_view(), stream(), br->device_mr());
                    packed_data = std::make_unique<PackedData>(
                        std::move(packed_columns.metadata),
                        br->move(std::move(packed_columns.gpu_data), stream())
                    );

                    // Handle the case where `cudf::pack` allocates slightly more than the
                    // input size. This can occur because cudf uses aligned allocations,
                    // which may exceed the requested size. To accommodate this, we
                    // allow some wiggle room.
                    if (packed_data->data->size > reservation.size()) {
                        auto const wiggle_room =
                            1024 * static_cast<std::size_t>(table_view().num_columns());
                        if (packed_data->data->size <= reservation.size() + wiggle_room) {
                            reservation = br->reserve(
                                                MemoryType::HOST,
                                                packed_data->data->size,
                                                AllowOverbooking::YES
                            )
                                              .first;
                        }
                    }
                    packed_data->data =
                        br->move(std::move(packed_data->data), reservation);
                }
                return TableChunk(std::move(packed_data));
            }
        default:
            RAPIDSMPF_FAIL("MemoryType: unknown");
        }
    }
    RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "something went wrong");

    auto metadata = std::make_unique<std::vector<std::uint8_t>>(*packed_data_->metadata);
    auto data =
        br->allocate(packed_data_->data->size, packed_data_->stream(), reservation);
    buffer_copy(*data, *packed_data_->data, packed_data_->data->size);
    return TableChunk(std::make_unique<PackedData>(std::move(metadata), std::move(data)));
}

ContentDescription get_content_description(TableChunk const& obj) {
    ContentDescription ret{
        obj.is_spillable() ? ContentDescription::Spillable::YES
                           : ContentDescription::Spillable::NO
    };
    for (auto mem_type : MEMORY_TYPES) {
        ret.content_size(mem_type) = obj.data_alloc_size(mem_type);
    }
    return ret;
}

Message to_message(std::uint64_t sequence_number, std::unique_ptr<TableChunk> chunk) {
    auto cd = get_content_description(*chunk);
    return Message{
        sequence_number,
        std::move(chunk),
        cd,
        [](Message const& msg, MemoryReservation& reservation) -> Message {
            auto const& self = msg.get<TableChunk>();
            auto chunk = std::make_unique<TableChunk>(self.copy(reservation));
            auto cd = get_content_description(*chunk);
            return Message{msg.sequence_number(), std::move(chunk), cd, msg.copy_cb()};
        }
    };
}

}  // namespace rapidsmpf::streaming
