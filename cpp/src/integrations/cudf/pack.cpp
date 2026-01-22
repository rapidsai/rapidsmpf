/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/span.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/pack.hpp>

namespace rapidsmpf {

TablePacker::TablePacker(
    std::size_t bounce_buffer_size, rmm::cuda_stream_view alloc_stream, BufferResource& br
)
    : bounce_buf_(std::make_unique<rmm::device_buffer>(
          bounce_buffer_size, alloc_stream, br.device_mr()
      )) {}

TablePacker::PackOp TablePacker::aquire(
    cudf::table_view const& input,
    rmm::cuda_stream_view pack_stream,
    rmm::device_async_resource_ref pack_temp_mr
) {
    return TablePacker::PackOp(input, pack_stream, pack_temp_mr, *this);
}

TablePacker::PackOp::PackOp(
    cudf::table_view const& input,
    rmm::cuda_stream_view pack_stream,
    rmm::device_async_resource_ref pack_temp_mr,
    TablePacker& table_packer
)
    : lock_(table_packer.mutex_),
      bounce_buffer_size_(table_packer.bounce_buf_->size()),
      pack_stream_(pack_stream),
      cpack_(cudf::chunked_pack::create(
          input, bounce_buffer_size_, pack_stream, pack_temp_mr
      )),
      table_packer_(table_packer) {}

TablePacker::PackOp::~PackOp() {
    clear();
}

std::size_t TablePacker::PackOp::get_packed_size() const {
    return cpack_->get_total_contiguous_size();
}

std::unique_ptr<std::vector<uint8_t>> TablePacker::PackOp::build_metadata() const {
    return cpack_->build_metadata();
}

std::size_t TablePacker::PackOp::pack(Buffer& dest_buf) {
    RAPIDSMPF_EXPECTS(
        dest_buf.size >= get_packed_size(),
        "Destination buffer is too small",
        std::invalid_argument
    );

    if (is_device_accessible(dest_buf.mem_type())) {
        return pack_by_dest_buf_offset(dest_buf);
    } else {
        return pack_by_copying(dest_buf);
    }
}

std::size_t TablePacker::PackOp::pack_by_dest_buf_offset(Buffer& dest_buf) {
    // directly pack into the destination buffer. This requires dest_buf memory type to be
    // device accessible.
    lock_.unlock();  // release the lock on the bounce buffer.

    size_t const packed_size = get_packed_size();

    dest_buf.write_access([&](std::byte* dest_ptr,
                              rmm::cuda_stream_view dest_buf_stream) {
        cuda_stream_join(pack_stream_, dest_buf_stream);

        std::size_t off = 0;
        while (cpack_->has_next()) {
            // TODO: in the last iteration, this device span will go beyond the size of
            // the dest_buf, but it should not copy beyond the size of the dest_buf.
            // We can't reduce the size of the device span, because the chunked_pack
            // requires it to be bounce_buffer_size_ always. Could cause a
            // compute-sanitizer error?
            // Possible solution is to create a new chunked_pack with dest_buf size and
            // pack in one go.
            off += cpack_->next(cudf::device_span<uint8_t>(
                reinterpret_cast<uint8_t*>(dest_ptr) + off, bounce_buffer_size_
            ));
        }
        RAPIDSMPF_EXPECTS(off == packed_size, "Packed size mismatch");

        // Synchronize pack_stream_ with dest_buf's stream before returning.
        cuda_stream_join(dest_buf_stream, pack_stream_);
    });

    return packed_size;
}

std::size_t TablePacker::PackOp::pack_by_copying(Buffer& dest_buf) {
    std::size_t offset =
        dest_buf.write_access([&](std::byte* dest_ptr,
                                  rmm::cuda_stream_view dest_buf_stream) {
            cuda_stream_join(pack_stream_, dest_buf_stream);

            std::size_t off = 0;
            cudf::device_span<uint8_t> bounce_buf_span(
                static_cast<std::uint8_t*>(table_packer_.bounce_buf_->data()),
                table_packer_.bounce_buf_->size()
            );
            while (cpack_->has_next()) {
                auto const bytes_copied = cpack_->next(bounce_buf_span);
                RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                    dest_ptr + off,
                    bounce_buf_span.data(),
                    bytes_copied,
                    cudaMemcpyDefault,
                    pack_stream_.value()
                ));
                off += bytes_copied;
            }
            // Synchronize pack_stream_ with dest_buf's stream before returning.
            cuda_stream_join(dest_buf_stream, pack_stream_);
            return off;
        });

    // Update bounce buffer's stream so the next pack operation can safely reuse it.
    table_packer_.bounce_buf_->set_stream(pack_stream_);

    return offset;
}

void TablePacker::PackOp::clear() {
    cpack_.reset();
    if (lock_.owns_lock()) {
        lock_.unlock();
    }
}

}  // namespace rapidsmpf
