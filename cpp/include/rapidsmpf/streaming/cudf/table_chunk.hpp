/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <memory>
#include <optional>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rapidsmpf/buffer/content_description.hpp>
#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/owning_wrapper.hpp>

namespace rapidsmpf::streaming {


/**
 * @brief A unit of table data in a streaming pipeline.
 *
 * Represents either an unpacked `cudf::table`, a `cudf::packed_columns`, or a
 * `PackedData`.
 *
 * TableChunks may be initially unavailable (e.g., if the data is packed or spilled),
 * and can be made available (i.e., materialized to device memory) on demand.
 */
class TableChunk {
  public:
    /**
     * @brief Indicates whether the TableChunk holds an exclusive or shared view
     * of the underlying table data.
     *
     * This boolean enum is used to explicitly express ownership semantics
     * when constructing a TableChunk from a `cudf::table_view`.
     *
     * - `ExclusiveView::YES`: The TableChunk has exclusive ownership of
     *   the table's device memory and are considered spillable.
     *
     * - `ExclusiveView::NO`: The TableChunk is a non-owning view of data
     *   managed elsewhere. The memory may be shared or externally owned,
     *   and the chunk is therefore not spillable.
     */
    enum class ExclusiveView : bool {
        NO,
        YES,
    };

    /**
     * @brief Construct a TableChunk from a device table.
     *
     * @param table Device-resident table.
     * @param stream The CUDA stream on which the table was created.
     */
    TableChunk(std::unique_ptr<cudf::table> table, rmm::cuda_stream_view stream);

    /**
     * @brief Construct a TableChunk from a device table view.
     *
     * The TableChunk does not take ownership of the underlying data; instead, the
     * provided @p owner object is kept alive for the lifetime of the TableChunk.
     * The caller is responsible for ensuring that the underlying device memory
     * referenced by @p table_view remains valid during this period.
     *
     * This constructor is typically used when creating a TableChunk from Python,
     * where @p owner is used to keep the corresponding Python object alive until
     * the TableChunk is destroyed.
     *
     * @param table_view Device-resident table view.
     * @param device_alloc_size Number of bytes allocated in device memory.
     * @param stream CUDA stream on which the table was created.
     * @param owner Object owning the memory backing @p table_view. This object will be
     * destroyed last when the TableChunk is destroyed or spilled.
     * @param exclusive_view Specifies whether this TableChunk has exclusive ownership
     * semantics over the underlying table data:
     *   - When `ExclusiveView::YES`, the following guarantees must hold:
     *       - The @p table_view is the sole representation of the table.
     *       - The @p owner exclusively owns the table memory.
     *     These guarantees allow the TableChunk to be spillable and ensure that
     *     destroying @p owner will correctly free the associated device memory.
     *   - When `ExclusiveView::NO`, the chunk is considered a non-owning view and
     *     is therefore not spillable.
     */
    TableChunk(
        cudf::table_view table_view,
        std::size_t device_alloc_size,
        rmm::cuda_stream_view stream,
        OwningWrapper&& owner,
        ExclusiveView exclusive_view
    );

    /**
     * @brief Construct a TableChunk from packed columns.
     *
     * @param packed_columns Serialized device table.
     * @param stream The CUDA stream on which the packed_columns was created.
     */
    TableChunk(
        std::unique_ptr<cudf::packed_columns> packed_columns, rmm::cuda_stream_view stream
    );

    /**
     * @brief Construct a TableChunk from a packed data blob.
     *
     * The packed data's CUDA stream will be associated the new table chunk.
     *
     * @param packed_data Serialized host/device data with metadata.
     */
    TableChunk(std::unique_ptr<PackedData> packed_data);

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
     * @brief Returns the CUDA stream on which this table chunk was created.
     *
     * @return The CUDA stream view.
     */
    [[nodiscard]] rmm::cuda_stream_view stream() const noexcept;

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
     * @brief Moves this table chunk into a new one with its cudf table made available.
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
     * @brief Indicates whether this table chunk can be spilled to device memory.
     *
     * A table chunk is considered spillable if it owns its underlying memory. This is
     * true when it was created from one of the following:
     *   - A device-owning source such as a `cudf::table`, `cudf::packed_columns`, or
     *     `PackedData`.
     *   - A `cudf::table_view` constructed with `is_exclusive_view == true`, indicating
     *     that the view is the sole representation of the underlying data and that its
     *     owner exclusively manages the table's memory.
     *
     * In contrast, chunks constructed from non-exclusive `cudf::table_view` instances are
     * non-owning views of externally managed memory and therefore not spillable.
     *
     * To spill a table chunk from device to host memory, first call `copy()` to create a
     * host-side copy, then delete or overwrite the original device chunk. If
     * `is_spillable() == true`, destroying the original device chunk will release the
     * associated device memory.
     *
     * @return `true` if the table chunk owns its memory and can be spilled; otherwise
     * `false`.
     */
    [[nodiscard]] bool is_spillable() const;

    /**
     * @brief Create a deep copy of the table chunk.
     *
     * Allocates new memory for all buffers in the table using the specified
     * `reservation`, which determines the target memory type (e.g., host or device).
     * As a consequence, the `is_available()` status may differ in the new copy. For
     * example, copying an available table chunk from device to host memory will result
     * in an unavailable copy.
     *
     * @param br Buffer resource used for allocations.
     * @param reservation Memory reservation used to track and limit allocations.
     * @return A new `TableChunk` instance containing copies of all buffers and metadata.
     *
     * @throws std::overflow_error If the total allocation size exceeds the available
     * reservation.
     */
    [[nodiscard]] TableChunk copy(
        BufferResource* br, MemoryReservation& reservation
    ) const;

  private:
    ///< @brief Optional owning object if the TableChunk was constructed from a
    ///< table_view.
    OwningWrapper owner_{};

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
    bool is_spillable_;
};

/**
 * @brief Generate a content description for a `TableChunk`.
 *
 * @param obj The object's content to describe.
 * @return A new content description.
 */
ContentDescription get_content_description(TableChunk const& obj);

/**
 * @brief Wrap a `TableChunk` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
Message to_message(std::uint64_t sequence_number, std::unique_ptr<TableChunk> chunk);

}  // namespace rapidsmpf::streaming
