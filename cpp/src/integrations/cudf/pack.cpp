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
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

namespace detail {

std::unique_ptr<PackedData> pack_device(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::DEVICE,
        "pack_device requires a device memory reservation",
        std::invalid_argument
    );
    auto packed_columns = cudf::pack(table, stream, reservation.br()->device_mr());
    reservation.br()->release(reservation, packed_columns.gpu_data->size());

    return std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        reservation.br()->move(std::move(packed_columns.gpu_data), stream)
    );
}

std::unique_ptr<PackedData> pack_pinned_host(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::PINNED_HOST,
        "pack_pinned_host requires a pinned host memory reservation",
        std::invalid_argument
    );
    // benchmarks shows that using cudf::pack with pinned mr is sufficient.

    auto packed_columns =
        cudf::pack(table, stream, reservation.br()->pinned_mr_as_device());
    reservation.br()->release(reservation, packed_columns.gpu_data->size());

    // packed table is now in rmm::device_buffer. We need to move it to a
    // HostBuffer as Pinned memory allocations are only used with HostBuffers.
    auto pinned_host_buffer =
        std::make_unique<HostBuffer>(HostBuffer::from_rmm_device_buffer(
            std::move(packed_columns.gpu_data), stream, reservation.br()->pinned_mr()
        ));
    return std::make_unique<PackedData>(
        std::move(packed_columns.metadata),
        reservation.br()->move(
            std::move(pinned_host_buffer), stream, MemoryType::PINNED_HOST
        )
    );
}

// Handle the case where packing allocates slightly more than the input size. This can
// occur because cudf uses aligned allocations, which may exceed the requested size. To
// accommodate this, we allow some wiggle room.
void resize_res_if_possible(
    size_t packed_size,
    cudf::size_type num_columns,
    MemoryReservation& reservation,
    BufferResource* br
) {
    if (packed_size > reservation.size()) {
        auto const wiggle_room = 1024 * static_cast<std::size_t>(num_columns);
        if (packed_size <= reservation.size() + wiggle_room) {
            reservation =
                br->reserve(reservation.mem_type(), packed_size, AllowOverbooking::YES)
                    .first;
        }
    }
}

std::unique_ptr<PackedData> pack_host(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    RAPIDSMPF_EXPECTS(
        reservation.mem_type() == MemoryType::HOST,
        "pack_host requires a host memory reservation",
        std::invalid_argument
    );

    auto br = reservation.br();

    if (table.num_rows() == 0) {  // special case for empty table
        auto empty_buf = br->allocate(0, stream, reservation);
        return std::make_unique<PackedData>(
            std::make_unique<std::vector<uint8_t>>(cudf::pack_metadata(
                table, reinterpret_cast<std::uint8_t const*>(empty_buf->data()), 0
            )),
            std::move(empty_buf)
        );
    }

    // minimum buffer size for chunked packing
    static constexpr size_t chunked_pack_min_buffer_size = 1024 * 1024;

    // Fast path for small tables (< 1MB): pack directly on device and copy to host.
    // This avoids the overhead of allocating a 1MB bounce buffer for chunked packing.
    auto const raw_est_size = estimated_memory_usage(table, stream);
    if (raw_est_size < chunked_pack_min_buffer_size) {
        auto packed_columns = cudf::pack(table, stream, br->device_mr());
        auto const packed_size = packed_columns.gpu_data->size();

        // Allocate host buffer from the host reservation
        detail::resize_res_if_possible(packed_size, table.num_columns(), reservation, br);
        auto dest_buf = br->allocate(packed_size, stream, reservation);
        dest_buf->write_access([&](std::byte* dest_ptr, rmm::cuda_stream_view) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                dest_ptr,
                packed_columns.gpu_data->data(),
                packed_size,
                cudaMemcpyDefault,
                stream
            ));
        });

        return std::make_unique<PackedData>(
            std::move(packed_columns.metadata), std::move(dest_buf)
        );
    }

    // estimate the size of the table with a minimum of 1MiB
    auto const est_size = std::max(raw_est_size, chunked_pack_min_buffer_size);

    // max available memory for bounce buffer
    auto max_available = [](size_t res_sz, size_t ob) -> size_t {
        return res_sz > ob ? res_sz - ob : 0;
    };

    {  // try device memory first
        auto [dev_res, dev_ob] =
            br->reserve(MemoryType::DEVICE, est_size, AllowOverbooking::YES);

        auto dev_avail = max_available(dev_res.size(), dev_ob);
        if (dev_avail >= chunked_pack_min_buffer_size) {
            rmm::device_buffer bounce_buffer(dev_avail, stream, br->device_mr());
            dev_res.clear();
            return chunked_pack(
                table, stream, bounce_buffer, br->device_mr(), reservation
            );
        }
    }

    if (br->is_pinned_memory_available()) {
        auto [pinned_res, pinned_ob] =
            br->reserve(MemoryType::PINNED_HOST, est_size, AllowOverbooking::YES);

        auto pinned_avail = max_available(pinned_res.size(), pinned_ob);
        if (pinned_avail >= chunked_pack_min_buffer_size) {
            rmm::device_buffer bounce_buffer(
                pinned_avail, stream, br->pinned_mr_as_device()
            );
            pinned_res.clear();
            return chunked_pack(
                table, stream, bounce_buffer, br->pinned_mr_as_device(), reservation
            );
        }
    }

    // all attempts failed.
    RAPIDSMPF_FAIL(
        "Failed to allocate bounce buffer for chunked packing to host memory.",
        std::runtime_error
    );
}

}  // namespace detail

std::unique_ptr<PackedData> pack(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream,
    MemoryReservation& reservation
) {
    switch (reservation.mem_type()) {
    case MemoryType::DEVICE:
        return detail::pack_device(table, stream, reservation);
    case MemoryType::PINNED_HOST:
        return detail::pack_pinned_host(table, stream, reservation);
    case MemoryType::HOST:
        return detail::pack_host(table, stream, reservation);
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

    detail::resize_res_if_possible(packed_size, table.num_columns(), reservation, br);

    auto dest_buf = br->allocate(packed_size, stream, reservation);

    if (packed_size > 0) {
        dest_buf->write_access([&](std::byte* dest_ptr,
                                   rmm::cuda_stream_view dest_buf_stream) {
            // join the destination stream with the bounce buffer stream so that the
            // copies can be queued
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
    }

    return std::make_unique<PackedData>(cpack.build_metadata(), std::move(dest_buf));
}

}  // namespace rapidsmpf
