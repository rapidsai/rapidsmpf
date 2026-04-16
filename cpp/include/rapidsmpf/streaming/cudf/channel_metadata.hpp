/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <cudf/types.hpp>

#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Hash partitioning scheme.
 *
 * Rows are distributed by `hash(columns[column_indices]) % modulus`.
 */
struct HashScheme {
    std::vector<cudf::size_type> column_indices;  ///< Column indices to hash on.
    int modulus;  ///< Hash modulus (number of partitions).

    /**
     * @brief Equality comparison.
     * @return True if both schemes are equal.
     */
    bool operator==(HashScheme const&) const = default;
};

/**
 * @brief A single sort key: column index, sort direction, and null placement.
 */
struct OrderKey {
    cudf::size_type column_index;  ///< Column to sort on.
    cudf::order order;  ///< ASCENDING or DESCENDING.
    cudf::null_order null_order;  ///< BEFORE or AFTER.

    /**
     * @brief Equality comparison.
     * @return True if all fields are equal.
     */
    bool operator==(OrderKey const&) const = default;
};

/**
 * @brief Order-based partitioning scheme for sorted/range-partitioned data.
 *
 * Data is partitioned by value ranges based on predetermined boundaries.
 * For N partitions, there are N-1 boundary rows:
 * - Partition 0: values < boundaries[0]
 * - Partition i (0 < i < N-1): boundaries[i-1] <= values < boundaries[i]
 * - Partition N-1: values >= boundaries[N-2]
 *
 * `keys[i]` is the i-th sort column; ordering is lexicographic by `keys[0]`,
 * then `keys[1]`, and so on.
 *
 * When `boundaries` is set, its columns must align with `keys`
 * (same count and compatible dtypes). Mismatched dtypes are a usage error.
 */
struct OrderScheme {
    std::vector<OrderKey> keys;  ///< Sort keys (column, order, null_order per entry).
    std::unique_ptr<TableChunk> boundaries;  ///< N-1 boundary rows for N partitions.
    bool strict_boundary{false};  ///< Sort keys disjoint across chunks.

    /**
     * @brief Deep-copy this scheme into a new one.
     *
     * Copies vectors and `strict_boundary`. If `boundaries` is non-null, copies
     * the underlying `TableChunk` via `TableChunk::copy(reservation)`.
     *
     * @param reservation Memory reservation for the boundary copy.
     * @return A new independent `OrderScheme`.
     */
    [[nodiscard]] OrderScheme clone(MemoryReservation& reservation) const;

    /**
     * @brief Shallow metadata equality (not semantic boundary value equality).
     *
     * Returns true when `keys` and `strict_boundary` match, and boundary tables
     * are consistent in the weak sense: both absent, or both present with the
     * same `(num_rows, num_columns)`.
     * Cell values inside `boundaries` are intentionally not compared (that would
     * require a device comparison API with stream and memory resource). Do not
     * use `operator==` to assert that two schemes have identical range boundaries.
     * A future API may add deep boundary comparison with explicit stream and MR.
     *
     * @param other The OrderScheme to compare against.
     * @return True under the shallow rules above.
     */
    bool operator==(OrderScheme const& other) const;
};

/**
 * @brief Partitioning specification for a single hierarchical level.
 *
 * Represents how data is partitioned at one level of the hierarchy
 * (e.g., inter-rank or local). Use the static factory methods to construct.
 *
 * - `none()`: No partitioning information at this level.
 * - `inherit()`: Partitioning is inherited from the parent level unchanged.
 * - `from_hash(h)`: Explicit hash partitioning with the given scheme.
 * - `from_order(o)`: Explicit order/range partitioning with the given scheme.
 */
struct PartitioningSpec {
    /**
     * @brief Type tag for PartitioningSpec.
     */
    enum class Type : std::uint8_t {
        NONE,  ///< No partitioning information at this level.
        INHERIT,  ///< Partitioning is inherited from parent level unchanged.
        HASH,  ///< Hash partitioning.
        ORDER,  ///< Order/range partitioning.
    };

    Type type = Type::NONE;  ///< The type of partitioning.
    std::optional<HashScheme> hash;  ///< Valid only when type == HASH.
    std::optional<OrderScheme> order;  ///< Valid only when type == ORDER.

    /**
     * @brief Create a spec indicating no partitioning information.
     * @return A PartitioningSpec with type NONE.
     */
    static PartitioningSpec none() {
        return {};
    }

    /**
     * @brief Create a spec indicating partitioning passes through from parent.
     * @return A PartitioningSpec with type INHERIT.
     */
    static PartitioningSpec inherit() {
        return {.type = Type::INHERIT, .hash = std::nullopt, .order = std::nullopt};
    }

    /**
     * @brief Create a spec for hash partitioning.
     * @param h The hash scheme to use.
     * @return A PartitioningSpec with type HASH.
     */
    static PartitioningSpec from_hash(HashScheme h) {
        return {.type = Type::HASH, .hash = std::move(h), .order = std::nullopt};
    }

    /**
     * @brief Create a spec for order/range partitioning.
     * @param o The order scheme to use. `o.keys` must be non-empty; otherwise
     * throws `std::invalid_argument`.
     * @return A PartitioningSpec with type ORDER.
     */
    static PartitioningSpec from_order(OrderScheme o);

    /**
     * @brief Deep-copy this spec, cloning any ORDER boundaries via `reservation`.
     *
     * NONE/INHERIT/HASH arms are trivially cheap; ORDER delegates to
     * `OrderScheme::clone(reservation)`.
     *
     * @param reservation Memory reservation forwarded to `OrderScheme::clone`.
     * @return A new independent `PartitioningSpec`.
     */
    [[nodiscard]] PartitioningSpec clone(MemoryReservation& reservation) const;

    /**
     * @brief Equality comparison.
     * @return True if both specs are equal.
     */
    bool operator==(PartitioningSpec const&) const = default;
};

/**
 * @brief Hierarchical partitioning metadata for a data stream.
 *
 * Describes how data flowing through a channel is partitioned at multiple
 * levels of the system hierarchy. Each level corresponds to a communicator
 * used to shuffle data at that level:
 *
 * - `inter_rank`: Distribution across ranks, corresponding to the primary
 *   communicator (e.g., `Context::comm()`). Shuffle operations at this level
 *   move data between ranks.
 * - `local`: Distribution within a rank, corresponding to a single-rank
 *   communicator. Operations at this level repartition data locally without
 *   network communication.
 */
struct Partitioning {
    /// Distribution across ranks (corresponds to primary communicator).
    PartitioningSpec inter_rank;
    /// Distribution within a rank (corresponds to local/single communicator).
    PartitioningSpec local;

    /**
     * @brief Deep-copy, cloning any ORDER boundaries via `reservation`.
     * @param reservation Memory reservation forwarded through the clone chain.
     * @return A new independent `Partitioning`.
     */
    [[nodiscard]] Partitioning clone(MemoryReservation& reservation) const;

    /**
     * @brief Equality comparison.
     * @return True if both partitionings are equal.
     */
    bool operator==(Partitioning const&) const = default;
};

/**
 * @brief Channel-level metadata describing the data stream.
 *
 * Contains information about chunk counts, partitioning, and duplication
 * status for the data flowing through a channel.
 */
struct ChannelMetadata {
    std::uint64_t local_count{};  ///< Local chunk-count estimate for this rank.
    Partitioning partitioning;  ///< How the data is partitioned.
    bool duplicated{};  ///< Whether data is duplicated on all workers.

    /// @brief Default constructor.
    ChannelMetadata() = default;

    /**
     * @brief Construct metadata with specified values.
     *
     * @param local_count Local chunk count.
     * @param partitioning Partitioning metadata (default: no partitioning).
     * @param duplicated Whether data is duplicated (default: false).
     */
    ChannelMetadata(
        std::uint64_t local_count, Partitioning partitioning = {}, bool duplicated = false
    )
        : local_count(local_count),
          partitioning(std::move(partitioning)),
          duplicated(duplicated) {}

    /**
     * @brief Deep-copy, cloning any ORDER boundaries via `reservation`.
     * @param reservation Memory reservation forwarded through the clone chain.
     * @return A new independent `ChannelMetadata`.
     */
    [[nodiscard]] ChannelMetadata clone(MemoryReservation& reservation) const;

    /**
     * @brief Equality comparison.
     * @return True if both metadata objects are equal.
     */
    bool operator==(ChannelMetadata const&) const = default;
};

/**
 * @brief Compute a `ContentDescription` for a `ChannelMetadata`.
 *
 * Walks `m.partitioning.inter_rank` and `m.partitioning.local` and accumulates
 * per-memory-type sizes from any ORDER boundary `TableChunk`. Spillability is
 * set to `Spillable::YES`; the copy callback in `to_message` handles both
 * device-to-device and device-to-host (spill) paths via `clone(reservation)`.
 *
 * @param m The metadata to describe.
 * @return A `ContentDescription` reflecting boundary device bytes.
 */
ContentDescription content_description_for(ChannelMetadata const& m);

/**
 * @brief Wrap a `ChannelMetadata` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param m The metadata to wrap.
 * @return A `Message` encapsulating the metadata as its payload.
 */
Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m);

}  // namespace rapidsmpf::streaming
