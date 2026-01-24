/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

using namespace rapidsmpf::streaming;

class StreamingPartitioning : public ::testing::Test {};

// ============================================================================
// HashScheme Tests
// ============================================================================

TEST_F(StreamingPartitioning, HashSchemeConstruction) {
    HashScheme h{{"col_a", "col_b"}, 16};
    EXPECT_EQ(h.columns.size(), 2);
    EXPECT_EQ(h.columns[0], "col_a");
    EXPECT_EQ(h.columns[1], "col_b");
    EXPECT_EQ(h.modulus, 16);
}

TEST_F(StreamingPartitioning, HashSchemeEquality) {
    HashScheme h1{{"key"}, 16};
    HashScheme h2{{"key"}, 16};
    HashScheme h3{{"key"}, 32};
    HashScheme h4{{"other"}, 16};

    EXPECT_EQ(h1, h2);
    EXPECT_NE(h1, h3);
    EXPECT_NE(h1, h4);
}

// ============================================================================
// PartitioningSpec Tests
// ============================================================================

TEST_F(StreamingPartitioning, PartitioningSpecNone) {
    auto spec = PartitioningSpec::none();
    EXPECT_TRUE(spec.is_none());
    EXPECT_FALSE(spec.is_aligned());
    EXPECT_FALSE(spec.is_hash());
    EXPECT_EQ(spec.type, SpecType::NONE);
}

TEST_F(StreamingPartitioning, PartitioningSpecAligned) {
    auto spec = PartitioningSpec::aligned();
    EXPECT_FALSE(spec.is_none());
    EXPECT_TRUE(spec.is_aligned());
    EXPECT_FALSE(spec.is_hash());
    EXPECT_EQ(spec.type, SpecType::ALIGNED);
}

TEST_F(StreamingPartitioning, PartitioningSpecHash) {
    auto spec = PartitioningSpec::from_hash(HashScheme{{"key"}, 16});
    EXPECT_FALSE(spec.is_none());
    EXPECT_FALSE(spec.is_aligned());
    EXPECT_TRUE(spec.is_hash());
    EXPECT_EQ(spec.type, SpecType::HASH);
    EXPECT_TRUE(spec.hash.has_value());
    EXPECT_EQ(spec.hash->columns[0], "key");
    EXPECT_EQ(spec.hash->modulus, 16);
}

TEST_F(StreamingPartitioning, PartitioningSpecEquality) {
    auto none1 = PartitioningSpec::none();
    auto none2 = PartitioningSpec::none();
    auto aligned1 = PartitioningSpec::aligned();
    auto aligned2 = PartitioningSpec::aligned();
    auto hash1 = PartitioningSpec::from_hash(HashScheme{{"key"}, 16});
    auto hash2 = PartitioningSpec::from_hash(HashScheme{{"key"}, 16});
    auto hash3 = PartitioningSpec::from_hash(HashScheme{{"key"}, 32});

    EXPECT_EQ(none1, none2);
    EXPECT_EQ(aligned1, aligned2);
    EXPECT_EQ(hash1, hash2);
    EXPECT_NE(none1, aligned1);
    EXPECT_NE(none1, hash1);
    EXPECT_NE(aligned1, hash1);
    EXPECT_NE(hash1, hash3);
}

// ============================================================================
// Partitioning Tests
// ============================================================================

TEST_F(StreamingPartitioning, PartitioningDefaultConstruction) {
    Partitioning p{};
    EXPECT_TRUE(p.inter_rank.is_none());
    EXPECT_TRUE(p.local.is_none());
}

TEST_F(StreamingPartitioning, PartitioningDirectGlobalShuffle) {
    // Direct global shuffle to N_g partitions: inter_rank=Hash, local=Aligned
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    EXPECT_TRUE(p.inter_rank.is_hash());
    EXPECT_TRUE(p.local.is_aligned());
    EXPECT_EQ(p.inter_rank.hash->modulus, 16);
}

TEST_F(StreamingPartitioning, PartitioningTwoStageShuffle) {
    // Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 4}),
        PartitioningSpec::from_hash(HashScheme{{"key"}, 8})
    };
    EXPECT_TRUE(p.inter_rank.is_hash());
    EXPECT_TRUE(p.local.is_hash());
    EXPECT_EQ(p.inter_rank.hash->modulus, 4);
    EXPECT_EQ(p.local.hash->modulus, 8);
}

TEST_F(StreamingPartitioning, PartitioningAfterLocalRepartition) {
    // After local repartition: inter_rank=Hash, local=None
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::none()
    };
    EXPECT_TRUE(p.inter_rank.is_hash());
    EXPECT_TRUE(p.local.is_none());
}

TEST_F(StreamingPartitioning, PartitioningEquality) {
    Partitioning p1{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    Partitioning p2{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    Partitioning p3{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 32}), PartitioningSpec::aligned()
    };

    EXPECT_EQ(p1, p2);
    EXPECT_NE(p1, p3);
}

// ============================================================================
// ChannelMetadata Tests
// ============================================================================

TEST_F(StreamingPartitioning, ChannelMetadataConstruction) {
    ChannelMetadata m{4, 16, Partitioning{}, false};
    EXPECT_EQ(m.local_count, 4);
    EXPECT_TRUE(m.global_count.has_value());
    EXPECT_EQ(m.global_count.value(), 16);
    EXPECT_FALSE(m.duplicated);
}

TEST_F(StreamingPartitioning, ChannelMetadataNoGlobalCount) {
    ChannelMetadata m{4};
    EXPECT_EQ(m.local_count, 4);
    EXPECT_FALSE(m.global_count.has_value());
    EXPECT_FALSE(m.duplicated);
}

TEST_F(StreamingPartitioning, ChannelMetadataWithPartitioning) {
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    ChannelMetadata m{4, 16, p, false};
    EXPECT_EQ(m.partitioning, p);
}

TEST_F(StreamingPartitioning, ChannelMetadataDuplicated) {
    ChannelMetadata m{1, 1, Partitioning{}, true};
    EXPECT_TRUE(m.duplicated);
}

TEST_F(StreamingPartitioning, ChannelMetadataEquality) {
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    ChannelMetadata m1{4, 16, p, false};
    ChannelMetadata m2{4, 16, p, false};
    ChannelMetadata m3{8, 16, p, false};

    EXPECT_EQ(m1, m2);
    EXPECT_NE(m1, m3);
}

// ============================================================================
// Message Integration Tests
// ============================================================================

TEST_F(StreamingPartitioning, PartitioningToMessage) {
    auto p = std::make_unique<Partitioning>(Partitioning{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    });
    Partitioning expected = *p;

    auto msg = to_message(42, std::move(p));
    EXPECT_FALSE(msg.empty());
    EXPECT_EQ(msg.sequence_number(), 42);
    EXPECT_TRUE(msg.holds<Partitioning>());

    auto const& got = msg.get<Partitioning>();
    EXPECT_EQ(got, expected);
}

TEST_F(StreamingPartitioning, PartitioningMessageRelease) {
    auto p = std::make_unique<Partitioning>(Partitioning{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    });
    Partitioning expected = *p;

    auto msg = to_message(0, std::move(p));
    auto got = msg.release<Partitioning>();
    EXPECT_TRUE(msg.empty());
    EXPECT_EQ(got, expected);
}

TEST_F(StreamingPartitioning, ChannelMetadataToMessage) {
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    auto m = std::make_unique<ChannelMetadata>(4, 16, p, false);
    ChannelMetadata expected = *m;

    auto msg = to_message(42, std::move(m));
    EXPECT_FALSE(msg.empty());
    EXPECT_EQ(msg.sequence_number(), 42);
    EXPECT_TRUE(msg.holds<ChannelMetadata>());

    auto const& got = msg.get<ChannelMetadata>();
    EXPECT_EQ(got, expected);
}

TEST_F(StreamingPartitioning, ChannelMetadataMessageRelease) {
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    auto m = std::make_unique<ChannelMetadata>(4, 16, p, false);
    ChannelMetadata expected = *m;

    auto msg = to_message(0, std::move(m));
    auto got = msg.release<ChannelMetadata>();
    EXPECT_TRUE(msg.empty());
    EXPECT_EQ(got, expected);
}
