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

constexpr size_t cudf_chunked_pack_min_buffer_size = 1024 * 1024;  // 1MB

/**
 * @brief Packing to host memory uses chunked packing with a bounce buffer.
 *
 * Algorithm:
 * 1. Special case: empty tables return immediately with empty packed data.
 * 2. Estimate the table size (est_size), with a minimum of 1MB.
 * 3. Try to reserve device memory for est_size with overbooking allowed.
 * 4. If available device memory (reservation - overbooking) >= 1MB,
 *    use chunked packing with the device bounce buffer.
 * 5. Otherwise, if pinned memory is available, retry with pinned memory (steps 3-4).
 * 6. If all attempts fail, throw an error.
 */
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
    // special case for empty table
    if (table.num_rows() == 0) {
        cudf::chunked_pack cpack(
            table, cudf_chunked_pack_min_buffer_size, stream, br->device_mr()
        );
        RAPIDSMPF_EXPECTS(
            cpack.get_total_contiguous_size() == 0,
            "empty table should have 0 contiguous size",
            std::runtime_error
        );
        return std::make_unique<PackedData>(
            cpack.build_metadata(), br->allocate(0, stream, reservation)
        );
    }

    // estimate the size of the table with a minimum of 1MiB
    auto const est_size = std::max(
        estimated_memory_usage(table, stream), cudf_chunked_pack_min_buffer_size
    );

    // max available memory for bounce buffer
    auto max_availble = [](size_t res_sz, size_t ob) -> size_t {
        return res_sz > ob ? res_sz - ob : 0;
    };

    // try device memory first
    auto [dev_res, dev_ob] =
        br->reserve(MemoryType::DEVICE, est_size, AllowOverbooking::YES);

    auto dev_avail = max_availble(dev_res.size(), dev_ob);
    if (dev_avail >= cudf_chunked_pack_min_buffer_size) {
        rmm::device_buffer bounce_buffer(dev_avail, stream, br->device_mr());
        dev_res.clear();
        return chunked_pack(table, stream, bounce_buffer, br->device_mr(), reservation);
    }
    dev_res.clear();  // Release unused device reservation.

    if (br->is_pinned_memory_available()) {  // try pinned memory as fallback
        auto [pinned_res, pinned_ob] =
            br->reserve(MemoryType::PINNED_HOST, est_size, AllowOverbooking::YES);

        auto pinned_avail = max_availble(pinned_res.size(), pinned_ob);
        if (pinned_avail >= cudf_chunked_pack_min_buffer_size) {
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

    // Handle the case where packing allocates slightly more than the
    // input size. This can occur because cudf uses aligned allocations,
    // which may exceed the requested size. To accommodate this, we
    // allow some wiggle room.
    if (packed_size > reservation.size()) {
        auto const wiggle_room = 1024 * static_cast<std::size_t>(table.num_columns());
        if (packed_size <= reservation.size() + wiggle_room) {
            reservation =
                br->reserve(reservation.mem_type(), packed_size, AllowOverbooking::YES)
                    .first;
        }
    }

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
