/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <rapidsmpf/streaming/cudf/channel_metadata.hpp>

using namespace rapidsmpf::streaming;

class StreamingChannelMetadata : public ::testing::Test {};

TEST_F(StreamingChannelMetadata, HashScheme) {
    HashScheme h{{"col_a", "col_b"}, 16};
    EXPECT_EQ(h.columns.size(), 2);
    EXPECT_EQ(h.columns[0], "col_a");
    EXPECT_EQ(h.modulus, 16);

    // Equality
    EXPECT_EQ(h, (HashScheme{{"col_a", "col_b"}, 16}));
    EXPECT_NE(h, (HashScheme{{"col_a", "col_b"}, 32}));
    EXPECT_NE(h, (HashScheme{{"other"}, 16}));
}

TEST_F(StreamingChannelMetadata, PartitioningSpec) {
    // None
    auto spec_none = PartitioningSpec::none();
    EXPECT_TRUE(spec_none.is_none());
    EXPECT_FALSE(spec_none.is_aligned());
    EXPECT_FALSE(spec_none.is_hash());

    // Aligned
    auto spec_aligned = PartitioningSpec::aligned();
    EXPECT_TRUE(spec_aligned.is_aligned());
    EXPECT_FALSE(spec_aligned.is_none());

    // Hash
    auto spec_hash = PartitioningSpec::from_hash(HashScheme{{"key"}, 16});
    EXPECT_TRUE(spec_hash.is_hash());
    EXPECT_EQ(spec_hash.hash->columns[0], "key");
    EXPECT_EQ(spec_hash.hash->modulus, 16);

    // Equality
    EXPECT_EQ(spec_none, PartitioningSpec::none());
    EXPECT_EQ(spec_aligned, PartitioningSpec::aligned());
    EXPECT_EQ(spec_hash, PartitioningSpec::from_hash(HashScheme{{"key"}, 16}));
    EXPECT_NE(spec_none, spec_aligned);
    EXPECT_NE(spec_hash, PartitioningSpec::from_hash(HashScheme{{"key"}, 32}));
}

TEST_F(StreamingChannelMetadata, PartitioningScenarios) {
    // Default construction
    Partitioning p_default{};
    EXPECT_TRUE(p_default.inter_rank.is_none());
    EXPECT_TRUE(p_default.local.is_none());

    // Direct global shuffle: inter_rank=Hash, local=Aligned
    Partitioning p_global{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    EXPECT_TRUE(p_global.inter_rank.is_hash());
    EXPECT_TRUE(p_global.local.is_aligned());
    EXPECT_EQ(p_global.inter_rank.hash->modulus, 16);

    // Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    Partitioning p_twostage{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 4}),
        PartitioningSpec::from_hash(HashScheme{{"key"}, 8})
    };
    EXPECT_EQ(p_twostage.inter_rank.hash->modulus, 4);
    EXPECT_EQ(p_twostage.local.hash->modulus, 8);

    // Equality
    EXPECT_EQ(
        p_global,
        (Partitioning{
            PartitioningSpec::from_hash(HashScheme{{"key"}, 16}),
            PartitioningSpec::aligned()
        })
    );
    EXPECT_NE(p_global, p_twostage);
}

TEST_F(StreamingChannelMetadata, ChannelMetadata) {
    // Full construction
    Partitioning p{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    ChannelMetadata m{4, 16, p, true};
    EXPECT_EQ(m.local_count, 4);
    EXPECT_EQ(m.global_count.value(), 16);
    EXPECT_EQ(m.partitioning, p);
    EXPECT_TRUE(m.duplicated);

    // Minimal construction (no global_count)
    ChannelMetadata m_minimal{4};
    EXPECT_EQ(m_minimal.local_count, 4);
    EXPECT_FALSE(m_minimal.global_count.has_value());
    EXPECT_FALSE(m_minimal.duplicated);

    // Equality
    EXPECT_EQ(m, (ChannelMetadata{4, 16, p, true}));
    EXPECT_NE(m, (ChannelMetadata{8, 16, p, true}));
}

TEST_F(StreamingChannelMetadata, MessageRoundTrip) {
    // Partitioning round-trip
    auto p = std::make_unique<Partitioning>(Partitioning{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    });
    Partitioning p_expected = *p;
    auto msg_p = to_message(42, std::move(p));
    EXPECT_EQ(msg_p.sequence_number(), 42);
    EXPECT_TRUE(msg_p.holds<Partitioning>());
    EXPECT_EQ(msg_p.release<Partitioning>(), p_expected);
    EXPECT_TRUE(msg_p.empty());

    // ChannelMetadata round-trip
    Partitioning part{
        PartitioningSpec::from_hash(HashScheme{{"key"}, 16}), PartitioningSpec::aligned()
    };
    auto m = std::make_unique<ChannelMetadata>(4, 16, part, false);
    ChannelMetadata m_expected = *m;
    auto msg_m = to_message(99, std::move(m));
    EXPECT_TRUE(msg_m.holds<ChannelMetadata>());
    EXPECT_EQ(msg_m.release<ChannelMetadata>(), m_expected);
    EXPECT_TRUE(msg_m.empty());
}
