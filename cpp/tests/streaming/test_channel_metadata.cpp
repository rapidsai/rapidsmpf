/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

// GCC 13.x/14.x have false positives on array-bounds and stringop-overflow when
// copying vectors through deeply inlined code paths (like std::optional copy
// constructors). Suppress for this file.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#include <gtest/gtest.h>

#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

using namespace rapidsmpf::streaming;

class StreamingChannelMetadata : public ::testing::Test {};

TEST_F(StreamingChannelMetadata, HashScheme) {
    HashScheme h{{0, 1}, 16};
    EXPECT_EQ(h.column_indices.size(), 2);
    EXPECT_EQ(h.column_indices[0], 0);
    EXPECT_EQ(h.column_indices[1], 1);
    EXPECT_EQ(h.modulus, 16);

    // Equality
    EXPECT_EQ(h, (HashScheme{{0, 1}, 16}));
    EXPECT_NE(h, (HashScheme{{0, 1}, 32}));
    EXPECT_NE(h, (HashScheme{{2}, 16}));
}

TEST_F(StreamingChannelMetadata, PartitioningSpec) {
    // None
    auto spec_none = PartitioningSpec::none();
    EXPECT_EQ(spec_none.type, PartitioningSpec::Type::NONE);

    // Passthrough
    auto spec_passthrough = PartitioningSpec::passthrough();
    EXPECT_EQ(spec_passthrough.type, PartitioningSpec::Type::PASSTHROUGH);

    // Hash
    auto spec_hash = PartitioningSpec::from_hash(HashScheme{{0}, 16});
    EXPECT_EQ(spec_hash.type, PartitioningSpec::Type::HASH);
    EXPECT_EQ(spec_hash.hash->column_indices[0], 0);
    EXPECT_EQ(spec_hash.hash->modulus, 16);

    // Equality
    EXPECT_EQ(spec_none, PartitioningSpec::none());
    EXPECT_EQ(spec_passthrough, PartitioningSpec::passthrough());
    EXPECT_EQ(spec_hash, PartitioningSpec::from_hash(HashScheme{{0}, 16}));
    EXPECT_NE(spec_none, spec_passthrough);
    EXPECT_NE(spec_hash, PartitioningSpec::from_hash(HashScheme{{0}, 32}));
}

TEST_F(StreamingChannelMetadata, PartitioningScenarios) {
    // Default construction
    Partitioning p_default{};
    EXPECT_EQ(p_default.inter_rank.type, PartitioningSpec::Type::NONE);
    EXPECT_EQ(p_default.local.type, PartitioningSpec::Type::NONE);

    // Direct global shuffle: inter_rank=Hash, local=Passthrough
    Partitioning p_global{
        PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::passthrough()
    };
    EXPECT_EQ(p_global.inter_rank.type, PartitioningSpec::Type::HASH);
    EXPECT_EQ(p_global.local.type, PartitioningSpec::Type::PASSTHROUGH);
    EXPECT_EQ(p_global.inter_rank.hash->modulus, 16);

    // Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    Partitioning p_twostage{
        PartitioningSpec::from_hash(HashScheme{{0}, 4}),
        PartitioningSpec::from_hash(HashScheme{{0}, 8})
    };
    EXPECT_EQ(p_twostage.inter_rank.hash->modulus, 4);
    EXPECT_EQ(p_twostage.local.hash->modulus, 8);

    // Equality
    EXPECT_EQ(
        p_global,
        (Partitioning{
            PartitioningSpec::from_hash(HashScheme{{0}, 16}),
            PartitioningSpec::passthrough()
        })
    );
    EXPECT_NE(p_global, p_twostage);
}

TEST_F(StreamingChannelMetadata, ChannelMetadata) {
    // Full construction
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::passthrough()
    };
    ChannelMetadata m{4, p, true};
    EXPECT_EQ(m.local_count, 4);
    EXPECT_EQ(m.partitioning, p);
    EXPECT_TRUE(m.duplicated);

    // Minimal construction
    ChannelMetadata m_minimal{4};
    EXPECT_EQ(m_minimal.local_count, 4);
    EXPECT_FALSE(m_minimal.duplicated);

    // Equality (avoid inline construction to prevent GCC 14.x false positive)
    ChannelMetadata m_same{4, p, true};
    ChannelMetadata m_diff{8, p, true};
    EXPECT_EQ(m, m_same);
    EXPECT_NE(m, m_diff);
}

TEST_F(StreamingChannelMetadata, MessageRoundTrip) {
    // ChannelMetadata round-trip (avoid copy to prevent GCC 14.x false positive)
    Partitioning part{
        PartitioningSpec::from_hash(HashScheme{{0}, 16}), PartitioningSpec::passthrough()
    };
    auto m = std::make_unique<ChannelMetadata>(4, part, false);
    auto msg_m = to_message(99, std::move(m));
    EXPECT_EQ(msg_m.sequence_number(), 99);
    EXPECT_TRUE(msg_m.holds<ChannelMetadata>());
    auto released = msg_m.release<ChannelMetadata>();
    EXPECT_EQ(released.local_count, 4);
    EXPECT_FALSE(released.duplicated);
    EXPECT_EQ(released.partitioning.inter_rank.hash->modulus, 16);
    EXPECT_TRUE(msg_m.empty());
}

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic pop
#endif
