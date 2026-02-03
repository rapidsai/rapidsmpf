/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/contiguous_split.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/pack.hpp>
#include <rapidsmpf/integrations/cudf/utils.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

namespace rapidsmpf {

namespace detail {

template <>
std::unique_ptr<PackedData> pack<MemoryType::DEVICE>(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::DEVICE,
        "pack<DEVICE> requires a device memory reservation",
        std::invalid_argument
    );
    auto packed_columns = cudf::pack(table, stream, reservation.br()->device_mr());
    reservation.br()->release(reservation, packed_columns.gpu_data->size());

    return std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        reservation.br()->move(std::move(packed_columns.gpu_data), stream)
    );
}

template <>
std::unique_ptr<PackedData> pack<MemoryType::PINNED_HOST>(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::PINNED_HOST,
        "pack<PINNED_HOST> requires a pinned host memory reservation",
        std::invalid_argument
    );
    // benchmarks shows that using cudf::pack with pinned mr is sufficient.

    auto packed_columns =
        cudf::pack(table, stream, reservation.br()->pinned_mr_as_device());
    reservation.br()->release(reservation, packed_columns.gpu_data->size());

    // packed table is now in rmm::device_buffer. We need to move it to a
    // HostBuffer as Pinned memory allocations are only used with HostBuffers.

    auto host_buffer = reservation.br()->move(std::move(packed_columns.gpu_data), stream);
    return std::make_unique<PackedData>(
        std::move(packed_columns.metadata), std::move(host_buffer)
    );
}

template <>
std::unique_ptr<PackedData> pack<MemoryType::HOST>(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::HOST,
        "pack<HOST> requires a host memory reservation",
        std::invalid_argument
    );

    auto br = reservation.br();
    // We use libcudf's pack() to serialize `table` into a
    // packed_columns and then we move the packed_columns' gpu_data to a
    // new host buffer.

    // TODO: use `cudf::chunked_pack()` with a bounce buffer. Currently,
    // `cudf::pack()` allocates device memory we haven't reserved.
    auto packed_columns = cudf::pack(table, stream, br->device_mr());
    auto packed_data = std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        br->move(std::move(packed_columns.gpu_data), stream)
    );

    // Handle the case where `cudf::pack` allocates slightly more than the
    // input size. This can occur because cudf uses aligned allocations,
    // which may exceed the requested size. To accommodate this, we
    // allow some wiggle room.
    if (packed_data->data->size > reservation.size()) {
        auto const wiggle_room = 1024 * static_cast<std::size_t>(table.num_columns());
        if (packed_data->data->size <= reservation.size() + wiggle_room) {
            reservation =
                br->reserve(
                      MemoryType::HOST, packed_data->data->size, AllowOverbooking::YES
                )
                    .first;
        }
    }
    packed_data->data = br->move(std::move(packed_data->data), reservation);
    return packed_data;
}

}  // namespace detail

std::unique_ptr<PackedData> pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    switch (reservation.mem_type()) {
    case MemoryType::DEVICE:
        return detail::pack<MemoryType::DEVICE>(table, stream, reservation);
    case MemoryType::PINNED_HOST:
        return detail::pack<MemoryType::PINNED_HOST>(table, stream, reservation);
    case MemoryType::HOST:
        return detail::pack<MemoryType::HOST>(table, stream, reservation);
    default:
        RAPIDSMPF_FAIL("unknown memory type");
    }
}

std::unique_ptr<PackedData> chunked_pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    rmm::device_buffer& bounce_buffer,
    rmm::device_async_resource_ref pack_temp_mr,
    MemoryReservation& reservation
) {
    auto br = reservation.br();

    cudf::chunked_pack cpack(table, bounce_buffer.size(), stream, pack_temp_mr);
    const auto packed_size = cpack.get_total_contiguous_size();

    auto dest_buf = br->allocate(packed_size, stream, reservation);

    dest_buf->write_access([&](std::byte* dest_ptr,
                               rmm::cuda_stream_view dest_buf_stream) {
        // join the destination stream with the bounce buffer stream so that the copies
        // can be queued
        cuda_stream_join(dest_buf_stream, bounce_buffer.stream());
        bounce_buffer.set_stream(dest_buf_stream);

        std::size_t off = 0;
        cudf::device_span<uint8_t> bounce_buf_span(
            static_cast<std::uint8_t*>(bounce_buffer.data()), bounce_buffer.size()
        );
        while (cpack.has_next()) {
            auto const bytes_copied = cpack.next(bounce_buf_span);
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                dest_ptr + off,
                bounce_buf_span.data(),
                bytes_copied,
                cudaMemcpyDefault,
                dest_buf_stream
            ));
            off += bytes_copied;
        }
    });

    return std::make_unique<PackedData>(cpack.build_metadata(), std::move(dest_buf));
}

}  // namespace rapidsmpf
