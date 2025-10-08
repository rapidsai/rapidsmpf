/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

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
     * @param stream The CUDA stream on which the table was created.
     */
    TableChunk(
        std::uint64_t sequence_number,
        std::unique_ptr<cudf::table> table,
        rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a TableChunk from a device table view.
     *
     * The TableChunk does not take ownership of the underlying data; the caller
     * is responsible for ensuring the data remains valid for the lifetime of
     * the TableChunk.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param table_view Device-resident table view.
     * @param device_alloc_size The number of bytes in device memory.
     * @param stream The CUDA stream on which the table was created.
     */
    TableChunk(
        std::uint64_t sequence_number,
        cudf::table_view table_view,
        std::size_t device_alloc_size,
        rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a TableChunk from packed columns.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param packed_columns Serialized device table.
     * @param stream The CUDA stream on which the packed_columns was created.
     */
    TableChunk(
        std::uint64_t sequence_number,
        std::unique_ptr<cudf::packed_columns> packed_columns,
        rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a TableChunk from a packed data blob.
     *
     * The packed data's CUDA stream will be associated the new table chunk.
     *
     * @param sequence_number Ordering identifier for the chunk.
     * @param packed_data Serialized host/device data with metadata.
     */
    TableChunk(std::uint64_t sequence_number, std::unique_ptr<PackedData> packed_data);

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
    [[nodiscard]] std::uint64_t sequence_number() const noexcept;

    /**
     * @brief Returns the CUDA stream on which this chunk was created.
     *
     * @return The CUDA stream view.
     */
    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept {
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
    [[nodiscard]] bool is_available() const noexcept;

    /**
     * @brief Returns the estimated cost (in bytes) of making the table available.
     *
     * Currently, only device memory cost is tracked.
     *
     * @return The cost in bytes.
     */
    [[nodiscard]] std::size_t make_available_cost() const noexcept;

    /**
     * @brief Moves this chunk into a new one with its cudf table made available.
     *
     * As part of the move, a copy or unpack may be performed, the associated CUDA
     * stream is used.
     *
     * @param reservation Memory reservation for allocations if needed.
     * @return A new TableChunk with data available on device.
     *
     * @note After this call, the current object is in a moved-from state;
     *       only reassignment, movement, or destruction are valid.
     */
    [[nodiscard]] TableChunk make_available(MemoryReservation& reservation);

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
     * Converts the device-resident table into a `PackedData` stored in host memory using
     * the associated CUDA stream.
     *
     * @param br Buffer resource used for allocations.
     * @return A new TableChunk containing packed host data.
     *
     * @note After this call, this object is in a has-been-moved-state and anything other
     * than reassignment, movement, and destruction is UB.
     */
    [[nodiscard]] TableChunk spill_to_host(BufferResource* br);

  private:
    std::uint64_t sequence_number_;

    // At most, one of the following unique pointers is non-null. If all of them are null,
    // the TableChunk is a non-owning view.
    // TODO: use a variant and drop the unique pointers?
    std::unique_ptr<cudf::table> table_;
    std::unique_ptr<cudf::packed_columns> packed_columns_;
    std::unique_ptr<PackedData> packed_data_;

    // Has value iff this TableChunk is available.
    std::optional<cudf::table_view> table_view_;

    // Zero initialized data allocation size (one for each memory type).
    std::array<std::size_t, MEMORY_TYPES.size()> data_alloc_size_ = {};
    std::size_t make_available_cost_;  // For now, only device memory cost is tracked.

    rmm::cuda_stream_view stream_;
};

}  // namespace rapidsmpf::streaming
