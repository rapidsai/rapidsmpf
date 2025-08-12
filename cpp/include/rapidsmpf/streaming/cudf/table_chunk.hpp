/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>

namespace rapidsmpf::streaming {


/**
 * @brief A unit of table data in a streaming pipeline.
 *
 * Represents either an unpacked `cudf::table`, a `cudf::packed_columns`, or a
 * `PackedData`, along with a sequence number to track chunk ordering.
 *
 * TableChunks may be initially unavailable (e.g., if the data is packed or spilled),
 * and can be made available (i.e., materialized to device memory) on demand.
 */
class TableChunk {
  public:
    /**
     * @brief Construct a TableChunk from a device table.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param table Device-resident table.
     * @param stream the CUDA stream on which the table was created.
     */
    TableChunk(
        std::uint64_t sequence_number,
        std::unique_ptr<cudf::table> table,
        rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a TableChunk from packed columns.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param packed_columns Serialized device table.
     * @param stream the CUDA stream on which the packed_columns was created.
     */
    TableChunk(
        std::uint64_t sequence_number,
        std::unique_ptr<cudf::packed_columns> packed_columns,
        rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a TableChunk from a packed data blob.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param packed_data Serialized host/device data with metadata.
     * @param stream the CUDA stream on which the packed_data was created.
     */
    TableChunk(
        std::uint64_t sequence_number,
        std::unique_ptr<PackedData> packed_data,
        rmm::cuda_stream_view stream
    );

    ~TableChunk() = default;

    /// @brief TableChunk is moveable
    TableChunk(TableChunk&&) = default;

    /**
     * @brief Move assignment
     *
     * @returns Moved this.
     */
    TableChunk& operator=(TableChunk&&) = default;
    TableChunk(TableChunk const&) = delete;
    TableChunk& operator=(TableChunk const&) = delete;

    /**
     * @brief Returns the sequence number of this chunk.
     * @return the sequence number.
     */
    [[nodiscard]] std::uint64_t sequence_number() const;

    /**
     * @brief Returns the CUDA stream on which this chunk was created.
     *
     * @return The CUDA stream view.
     */
    rmm::cuda_stream_view stream() {
        return stream_;
    }

    /**
     * @brief Number of bytes allocated for the data in the specified memory type.
     *
     * @param mem_type The memory type to query.
     * @return Number of bytes allocated.
     */
    [[nodiscard]] std::size_t data_alloc_size(MemoryType mem_type) const;

    /**
     * @brief Indicates whether the underlying cudf table data is fully available in
     * device memory.
     *
     * @return `true` if the table is already available; otherwise, `false`.
     */
    [[nodiscard]] bool is_available() const;

    /**
     * @brief Returns the estimated cost (in bytes) of making the table available.
     *
     * Currently, only device memory cost is tracked.
     *
     * @return The cost in bytes.
     */
    [[nodiscard]] std::size_t make_available_cost() const;

    /**
     * @brief Move this table chunk into a new table chunk where the underlying cudf table
     * is made available, possibly performing a copy or unpack.
     *
     * @param reservation Reservation used to allocate memory if needed.
     * @param stream CUDA stream to use for operations.
     * @param br Buffer resource used for allocations.
     * @return A new TableChunk with the data available on device.
     *
     * @note After this call, this object is in a has-been-moved-state and anything other
     * than reassignment, movement, and destruction is UB.
     */
    [[nodiscard]] TableChunk make_available(
        MemoryReservation& reservation, rmm::cuda_stream_view stream, BufferResource* br
    );

    /**
     * @brief Returns a view of the underlying table.
     *
     * The table must be available in device memory.
     *
     * @return cudf::table_view representing the table.
     *
     * @throws std::invalid_argument if `is_available() == false`.
     */
    [[nodiscard]] cudf::table_view table_view() const;

    /**
     * @brief Move this table chunk into host memory.
     *
     * Converts the device-resident table into a `PackedData` stored in host memory.
     *
     * @param stream CUDA stream to use for the copy.
     * @param br Buffer resource used for allocations.
     * @return A new TableChunk containing packed host data.
     *
     * @note After this call, this object is in a has-been-moved-state and anything other
     * than reassignment, movement, and destruction is UB.
     */
    [[nodiscard]] TableChunk spill_to_host(
        rmm::cuda_stream_view stream, BufferResource* br
    );

  private:
    std::uint64_t sequence_number_;

    // Only one of the following is non-null.
    // TODO: use a variant and drop the unique pointers?
    std::unique_ptr<cudf::table> table_;
    std::unique_ptr<cudf::packed_columns> packed_columns_;
    std::unique_ptr<PackedData> packed_data_;

    cudf::table_view table_view_;
    // Zero initialized data allocation size (one for each memory type).
    std::array<std::size_t, MEMORY_TYPES.size()> data_alloc_size_ = {};
    std::size_t make_available_cost_;  // For now, only device memory cost is tracked.

    rmm::cuda_stream_view stream_;
};

}  // namespace rapidsmpf::streaming
