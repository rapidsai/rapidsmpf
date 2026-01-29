/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <cudf/types.hpp>

#include <rapidsmpf/streaming/core/message.hpp>

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
 * @brief Partitioning specification for a single hierarchical level.
 *
 * Represents how data is partitioned at one level of the hierarchy
 * (e.g., inter-rank or local). Use the static factory methods to construct.
 *
 * - `none()`: No partitioning information at this level.
 * - `passthrough()`: Partitioning passes through from the parent level unchanged.
 * - `from_hash(h)`: Explicit hash partitioning with the given scheme.
 */
struct PartitioningSpec {
    /**
     * @brief Type tag for PartitioningSpec.
     */
    enum class Type : std::uint8_t {
        NONE,  ///< No partitioning information at this level.
        PASSTHROUGH,  ///< Partitioning passes through from parent level unchanged.
        HASH,  ///< Hash partitioning.
    };

    Type type = Type::NONE;  ///< The type of partitioning.
    std::optional<HashScheme> hash;  ///< Valid only when type == HASH.

    /**
     * @brief Create a spec indicating no partitioning information.
     * @return A PartitioningSpec with type NONE.
     */
    static PartitioningSpec none() {
        return {};
    }

    /**
     * @brief Create a spec indicating partitioning passes through from parent.
     * @return A PartitioningSpec with type PASSTHROUGH.
     */
    static PartitioningSpec passthrough() {
        return {.type = Type::PASSTHROUGH, .hash = std::nullopt};
    }

    /**
     * @brief Create a spec for hash partitioning.
     * @param h The hash scheme to use.
     * @return A PartitioningSpec with type HASH.
     */
    static PartitioningSpec from_hash(HashScheme h) {
        return {.type = Type::HASH, .hash = std::move(h)};
    }

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
     * @brief Equality comparison.
     * @return True if both metadata objects are equal.
     */
    bool operator==(ChannelMetadata const&) const = default;
};

/**
 * @brief Wrap a `ChannelMetadata` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param m The metadata to wrap.
 * @return A `Message` encapsulating the metadata as its payload.
 */
Message to_message(std::uint64_t sequence_number, std::unique_ptr<ChannelMetadata> m);

}  // namespace rapidsmpf::streaming
