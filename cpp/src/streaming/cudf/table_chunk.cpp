/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <filesystem>
#include <memory>
#include <sstream>

#include <cuda/cmath>

#include <cudf/contiguous_split.hpp>
#include <cudf/io/parquet.hpp>

#include <rapidsmpf/integrations/cudf/utils.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
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
        // table data is in device memory. We can trivially unpack it and make it
        // available.
        table_view_ = cudf::unpack(
            packed_data_->metadata->data(),
            reinterpret_cast<std::uint8_t const*>(packed_data_->data->data())
        );
        make_available_cost_ = 0;
    }
}

TableChunk::TableChunk(TableChunk&& other) noexcept
    : owner_(std::move(other.owner_)),
      table_(std::move(other.table_)),
      packed_data_(std::move(other.packed_data_)),
      table_view_(std::exchange(other.table_view_, std::nullopt)),
      data_alloc_size_(other.data_alloc_size_),
      make_available_cost_(other.make_available_cost_),
      stream_(other.stream_),
      is_spillable_(other.is_spillable_) {}

TableChunk& TableChunk::operator=(TableChunk&& other) noexcept {
    if (this != &other) {
        owner_ = std::move(other.owner_);
        table_ = std::move(other.table_);
        packed_data_ = std::move(other.packed_data_);
        table_view_ = std::exchange(other.table_view_, std::nullopt);
        data_alloc_size_ = other.data_alloc_size_;
        make_available_cost_ = other.make_available_cost_;
        stream_ = other.stream_;
        is_spillable_ = other.is_spillable_;
    }
    return *this;
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
    // Table chunk is not available. This means that the table data is not in device
    // memory. We need to move the table data to device memory using a device reservation.
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::DEVICE,
        "device memory reservation is required"
    );
    RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "packed data pointer cannot be null");
    auto packed_data = std::move(packed_data_);
    packed_data->data = reservation.br()->move(std::move(packed_data->data), reservation);
    return TableChunk{std::move(packed_data)};
}

TableChunk TableChunk::make_available(MemoryReservation&& reservation) {
    MemoryReservation& res = reservation;
    return make_available(res);
}

coro::task<TableChunk> TableChunk::make_available(
    std::shared_ptr<Context> ctx, std::int64_t net_memory_delta
) {
    co_return make_available(
        co_await reserve_memory(ctx, make_available_cost(), net_memory_delta)
    );
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
    // This method handles the three possible cases:
    //
    // 1. The chunk is available and the reservation specifies device memory.
    //    In this case, we can directly use cudf to create a deep copy of the
    //    table by copying the table_view() into device memory.
    //
    // 2. The chunk is available and the data is a generic cudf table that is
    //    not already packed. In this case, the table data must first be packed
    //    before copying it to host or pinned memory.
    //
    // 3. The chunk data is already packed (packed_data_ != nullptr).
    //    In this case, we simply use buffer_copy() to copy the packed data
    //    into the reservation-specified memory type. The original memory
    //    type of the chunk does not matter.
    BufferResource* br = reservation.br();
    if (is_available()) {
        switch (reservation.mem_type()) {
        case MemoryType::DEVICE:  // Case 1.
            {
                // Use libcudf to copy the table_view().
                auto const nbytes = data_alloc_size(MemoryType::DEVICE);
                StreamOrderedTiming timing{stream(), br->statistics()};
                auto table = std::make_unique<cudf::table>(
                    table_view(), stream(), br->device_mr()
                );
                br->statistics()->record_copy(
                    MemoryType::DEVICE, MemoryType::DEVICE, nbytes, std::move(timing)
                );
                // And update the provided `reservation`.
                br->release(reservation, nbytes);
                return TableChunk(std::move(table), stream());
            }
        case MemoryType::PINNED_HOST:
            if (packed_data_ == nullptr) {  // data is in device memory as a table
                size_t const block_size = br->access_pinned_mr().block_size();

                auto chunked_packer = cudf::chunked_pack(
                    table_view(), block_size, stream(), br->device_mr()
                );
                size_t const total_contiguous_size =
                    chunked_packer.get_total_contiguous_size();
                auto dest_buffer =
                    br->allocate(total_contiguous_size, stream(), reservation);

                size_t bytes_copied = 0;
                size_t count = 0;
                dest_buffer->write_access_blocks([&](std::span<std::byte> block,
                                                     rmm::cuda_stream_view /* stream */) {
                    count++;
                    if (!chunked_packer.has_next()) {
                        return;
                    }
                    cudf::device_span<std::uint8_t> device_span(
                        reinterpret_cast<std::uint8_t*>(block.data()), block.size()
                    );
                    bytes_copied += chunked_packer.next(device_span);
                });

                RAPIDSMPF_EXPECTS(
                    count == cuda::ceil_div(total_contiguous_size, block_size),
                    "count does not match total contiguous size"
                );

                if (bytes_copied != total_contiguous_size) {
                    auto const timestamp_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        )
                            .count();
                    std::ostringstream name_stream;
                    name_stream << "rapidsmpf_chunked_pack_debug_" << timestamp_ms
                                << "_bytes_" << bytes_copied << "_expected_"
                                << total_contiguous_size << ".parquet";
                    std::filesystem::path const debug_path =
                        std::filesystem::temp_directory_path() / name_stream.str();
                    cudf::io::sink_info sink{debug_path.string()};
                    auto const options =
                        cudf::io::parquet_writer_options::builder(sink, table_view())
                            .build();
                    cudf::io::write_parquet(options, stream());
                    RAPIDSMPF_FAIL(
                        "bytes copied (" + std::to_string(bytes_copied)
                            + ") does not match total contiguous size ("
                            + std::to_string(total_contiguous_size)
                            + "); table written to " + debug_path.string()
                            + " for verification (e.g. scripts/verify_chunked_pack_parquet.py)",
                        std::logic_error
                    );
                }

                return TableChunk(std::make_unique<PackedData>(
                    chunked_packer.build_metadata(), std::move(dest_buffer)
                ));
            }
            break;
        case MemoryType::HOST:
            // Case 2.
            if (packed_data_ == nullptr) {
                // We use libcudf's pack() to serialize `table_view()` into a
                // packed_columns and then we move the packed_columns' gpu_data to a
                // new host buffer.
                // TODO: use `cudf::chunked_pack()` with a bounce buffer. Currently,
                // `cudf::pack()` allocates device memory we haven't reserved.
                auto packed_columns = cudf::pack(table_view(), stream(), br->device_mr());
                auto packed_data = std::make_unique<PackedData>(
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
                                            reservation.mem_type(),
                                            packed_data->data->size,
                                            AllowOverbooking::YES
                        )
                                          .first;
                    }
                }
                packed_data->data = br->move(std::move(packed_data->data), reservation);
                return TableChunk(std::move(packed_data));
            }
            break;
        default:
            RAPIDSMPF_FAIL("MemoryType: unknown");
        }
    }
    // Note, `!is_available()` implies `packed_data_ != nullptr`.
    RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "something went wrong");

    // Case 3.
    auto const nbytes = packed_data_->data->size;
    auto metadata = std::make_unique<std::vector<std::uint8_t>>(*packed_data_->metadata);
    auto data = br->allocate(nbytes, packed_data_->stream(), reservation);
    buffer_copy(br->statistics(), *data, *packed_data_->data, nbytes);
    return TableChunk(std::make_unique<PackedData>(std::move(metadata), std::move(data)));
}

std::pair<cudf::size_type, cudf::size_type> TableChunk::shape() const noexcept {
    if (packed_data_ != nullptr) {
        auto view = cudf::packed_metadata_view(*packed_data_->metadata);
        return {view.num_rows(), view.num_columns()};
    }
    return {table_view_->num_rows(), table_view_->num_columns()};
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
