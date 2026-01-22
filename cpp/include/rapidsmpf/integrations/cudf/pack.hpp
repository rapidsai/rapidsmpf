/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <mutex>

#include <cudf/contiguous_split.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>

namespace rapidsmpf {

/**
 * @brief A thread-safe class for packing cudf tables using a bounce buffer.
 */
class TablePacker {
  public:
    /**
     * @brief Initialize a new table packer.
     * @param bounce_buffer_size the size of the bounce buffer
     * @param alloc_stream the stream to use for allocating the bounce buffer
     * @param br the buffer resource to use for allocating the bounce buffer
     */
    TablePacker(
        std::size_t bounce_buffer_size,
        rmm::cuda_stream_view alloc_stream,
        BufferResource& br
    );

    ~TablePacker() = default;

    TablePacker(TablePacker const&) = delete;
    TablePacker& operator=(TablePacker const&) = delete;
    TablePacker(TablePacker&&) = delete;
    TablePacker& operator=(TablePacker&&) = delete;

    /**
     * @brief A RAII wrapper for a pack operation.
     *
     * This class is used to wrap a pack operation and ensure that the resources are
     * released when the object is destroyed.
     */
    struct PackOp {
        friend class TablePacker;

      public:
        ~PackOp();

        /**
         * @brief get the size of the packed data
         * @return the size of the packed data
         */
        [[nodiscard]] std::size_t get_packed_size() const;

        /**
         * @brief build the metadata for the packed data
         * @return the metadata for the packed data
         */
        [[nodiscard]] std::unique_ptr<std::vector<uint8_t>> build_metadata() const;

        /**
         * @brief pack the data into the destination buffer.
         *
         * If the destination buffer is device accessible, this method will directly pack
         * the data into the destination buffer. Otherwise, it will iteratively copy data
         * from the bounce buffer to the destination buffer using cudaMemcpyAsync.
         *
         * @param dest_buf the destination buffer to pack the data into
         * @return the size of the packed data
         *
         * @throws std::invalid_argument if the destination buffer is too small
         */
        std::size_t pack(Buffer& dest_buf);

        /**
         * @brief clear the pack operation and release bounce buffer
         */
        void clear();

      private:
        /**
         * @brief initialize a new pack operation
         * @param input the input table to pack
         * @param pack_stream the stream to use for packing
         * @param pack_temp_mr the temporary memory resource to use for packing
         * @param table_packer the table packer to use
         */
        PackOp(
            cudf::table_view const& input,
            rmm::cuda_stream_view pack_stream,
            rmm::device_async_resource_ref pack_temp_mr,
            TablePacker& table_packer
        );

        /**
         * @brief This method will directly pack the data into the destination buffer.
         * This requires dest_buf memory type to be device accessible. `lock_` is released
         * before packing as the bounce buffer is not needed anymore.
         *
         * @param dest_buf the destination buffer to pack the data into
         * @return the size of the packed data
         */
        size_t pack_by_dest_buf_offset(Buffer& dest_buf);

        /**
         * @brief This method will iteratively copy data from the bounce buffer to the
         * destination buffer using cudaMemcpyAsync. Copies are ordered on the
         * `pack_stream_`. After packing, bounce buffer's stream will be set to
         * `pack_stream_` so that the next pack operation can safely reuse the bounce
         * buffer.
         * @param dest_buf the destination buffer to pack the data into
         * @return the size of the packed data
         */
        size_t pack_by_copying(Buffer& dest_buf);

        std::unique_lock<std::mutex> lock_;
        size_t const bounce_buffer_size_;
        rmm::cuda_stream_view pack_stream_;
        std::unique_ptr<cudf::chunked_pack> cpack_;
        TablePacker& table_packer_;
    };

    /**
     * @brief Aquire a pack operation to pack a cudf table.
     *
     * This method will block until the bounce buffer is available.
     *
     * @param input the input table to pack
     * @param pack_stream the stream to use for packing
     * @param pack_temp_mr the temporary memory resource to use for packing
     * @return a pack operation
     */
    PackOp aquire(
        cudf::table_view const& input,
        rmm::cuda_stream_view pack_stream,
        rmm::device_async_resource_ref pack_temp_mr
    );

  private:
    std::mutex mutex_;
    std::unique_ptr<rmm::device_buffer> bounce_buf_;
};

}  // namespace rapidsmpf
