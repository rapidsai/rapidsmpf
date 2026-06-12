/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/shuffler/finish_counter.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include "environment.hpp"
#include "utils.hpp"

extern Environment* GlobalEnvironment;

TEST(ReceivedChunks, spill_skips_control_messages) {
    auto mr = rmm::mr::get_current_device_resource_ref();
    auto br = rapidsmpf::BufferResource::create(mr);

    rapidsmpf::shuffler::detail::ReceivedChunks received;

    // Control messages have no data buffer (data_ == nullptr); spill must skip them
    // rather than calling data_memory_type(), which throws if data_ is null.
    received.insert(
        rapidsmpf::shuffler::detail::Chunk::from_finished_partition(
            /*chunk_id=*/0, /*part_id=*/0, /*expected_num_chunks=*/1
        )
    );

    EXPECT_EQ(received.spill(br.get(), /*amount=*/1024), 0UL);
}

TEST(ReceivedChunks, spill_respects_amount) {
    auto mr = rmm::mr::get_current_device_resource_ref();
    auto br = rapidsmpf::BufferResource::create(mr);
    auto stream = rmm::cuda_stream_default;

    rapidsmpf::shuffler::detail::ReceivedChunks received;
    constexpr std::size_t chunk_size = 100;

    for (rapidsmpf::shuffler::PartID pid = 0; pid < 2; ++pid) {
        auto metadata =
            std::make_unique<std::vector<std::uint8_t>>(std::size_t{1}, std::uint8_t{0});
        auto res = br->reserve_or_fail(chunk_size, rapidsmpf::MemoryType::DEVICE);
        auto data = br->make_buffer(chunk_size, stream, res);
        received.insert(
            rapidsmpf::shuffler::detail::Chunk::from_packed_data(
                0, pid, rapidsmpf::PackedData{std::move(metadata), std::move(data)}
            )
        );
    }

    // Two partitions, one 100-byte chunk each. spill() must stop after the first
    // partition satisfies the request; the outer loop must not continue into partition 1.
    EXPECT_EQ(received.spill(br.get(), chunk_size), chunk_size);
}

TEST(MetadataMessage, round_trip) {
    auto stream = rmm::cuda_stream_default;
    auto mr = rmm::mr::get_current_device_resource_ref();
    auto br = rapidsmpf::BufferResource::create(mr);

    auto metadata = iota_vector<std::uint8_t>(100);

    auto expect = rapidsmpf::shuffler::detail::Chunk::from_packed_data(
        1,  // chunk_id
        2,  // part_id
        rapidsmpf::PackedData{
            std::make_unique<std::vector<std::uint8_t>>(metadata),  // non-empty metadata
            br->move(std::make_unique<rmm::device_buffer>(), stream)  // empty data
        }
    );

    // Extract the metadata from then chunk.
    auto msg = expect.serialize();
    EXPECT_FALSE(expect.is_metadata_buffer_set());

    // Create a new chunk by deserializing the message.
    auto result = rapidsmpf::shuffler::detail::Chunk::deserialize(*msg, br.get());

    EXPECT_TRUE(expect.data_size() == 0 || result.is_data_buffer_set());
    // They should be identical.
    EXPECT_EQ(expect.part_id(), result.part_id());
    EXPECT_EQ(expect.chunk_id(), result.chunk_id());
    EXPECT_EQ(expect.expected_num_chunks(), result.expected_num_chunks());
    EXPECT_EQ(expect.data_size(), result.data_size());
    EXPECT_EQ(expect.metadata_size(), result.metadata_size());

    // The metadata should be identical to the original.
    EXPECT_EQ(metadata, *result.release_metadata_buffer());
}

namespace {

using MemoryLimitsMap = std::unordered_map<rapidsmpf::MemoryType, std::int64_t>;

/**
 * @brief Build a `memory_limits` map for a `BufferResource` that prioritizes one memory
 * type.
 *
 * All memory types are initialised to unlimited. If @p priorities is not
 * `MemoryType::DEVICE`, the device-memory limit is then set to zero, forcing
 * the `BufferResource` to allocate exclusively in host memory. Host memory is
 * never zeroed because it backs metadata and control-message allocations that
 * must always succeed.
 *
 * @param priorities The memory type to keep unlimited (all others are zeroed).
 * @return A map from each `MemoryType` to its byte limit (`std::int64_t`).
 */
MemoryLimitsMap get_memory_limits_map(rapidsmpf::MemoryType priorities) {
    using namespace rapidsmpf;

    // We set all memory types to be unlimited.
    MemoryLimitsMap ret = {
        {MemoryType::DEVICE, std::numeric_limits<std::int64_t>::max()},
        {MemoryType::HOST, std::numeric_limits<std::int64_t>::max()}
    };

    // And then set device memory to zero if it isn't prioritized.
    if (priorities != MemoryType::DEVICE) {
        ret.at(MemoryType::DEVICE) = 0;
    }
    // Note, we never set host memory to zero because it is used to allocate
    // stuff like metadata and control messages.
    return ret;
}

/// Conservation-preserving data model shared by the shuffler round-trip tests.
///
/// The index range `[0, total_num_rows)` is split into `total_num_partitions^2`
/// contiguous sub-regions via `chunk_indices` (front-loaded, so trailing sub-regions
/// are empty when `N < P*P`). Sub-region `(local_pidx, split_idx)` is piece
/// `k = local_pidx * P + split_idx` and is routed to destination partition
/// `split_idx`. The pieces exactly tile `[0, N)`, so the total shuffled row
/// count equals `N` regardless of rank or partition counts (conservation). A
/// per-shuffle `base` offset is added to every value so distinct shuffles carry
/// distinct data.


/**
 * @brief Build the input data for one owned partition region ready for insertion.
 *
 * Produces all non-empty sub-regions of the input region `local_pidx`, keyed by
 * their destination partition. Because `local_partitions()` across all ranks
 * partitions `[0, P)`, every input region is produced exactly once and rows are
 * never replicated.
 *
 * @param total_num_partitions Total number of shuffle partitions `P`.
 * @param total_num_rows       Total row count `N` tiled across all sub-regions.
 * @param local_pidx           Index of the locally-owned input region to generate.
 * @param stream               CUDA stream used for device allocations.
 * @param br                   Buffer resource used to allocate packed data.
 * @param base                 Offset added to every generated value (default 0).
 * @return Map from destination `PartID` to the corresponding `PackedData` chunk;
 *         empty sub-regions are omitted.
 */
std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>
make_partition_data(
    rapidsmpf::shuffler::PartID total_num_partitions,
    std::size_t total_num_rows,
    rapidsmpf::shuffler::PartID local_pidx,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br,
    std::int64_t base = 0
) {
    auto const P = static_cast<std::size_t>(total_num_partitions);
    auto const pieces = rapidsmpf::chunk_indices(total_num_rows, P * P);

    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
    for (rapidsmpf::shuffler::PartID split_idx = 0; split_idx < total_num_partitions;
         ++split_idx)
    {
        auto [start, end] = pieces[static_cast<std::size_t>(local_pidx) * P + split_idx];
        if (end > start) {
            chunks.emplace(
                split_idx,
                generate_packed_data(
                    end - start, base + static_cast<std::int64_t>(start), stream, br
                )
            );
        }
    }
    return chunks;
}

/**
 * @brief Verify that received chunks for a partition match the expected sub-regions.
 *
 * Recomputes the non-empty `(offset, count)` sub-regions expected for partition
 * `j` from the same conservation model used by `make_partition_data`, then
 * checks that @p received contains exactly those chunks (in any order). Chunks
 * are sorted by their embedded offset before comparison so the validation is
 * order-independent.
 *
 * @param received             Chunks extracted from the shuffler for partition `j`.
 * @param total_num_partitions Total number of shuffle partitions `P`.
 * @param total_num_rows       Total row count `N` used to tile sub-regions.
 * @param j                    Destination partition index being validated.
 * @param br                   Buffer resource used for unpacking received data.
 * @param base                 Offset that was added to every generated value (default 0).
 */
void validate_partition_data(
    std::vector<rapidsmpf::PackedData> received,
    rapidsmpf::shuffler::PartID total_num_partitions,
    std::size_t total_num_rows,
    rapidsmpf::shuffler::PartID j,
    rapidsmpf::BufferResource& br,
    std::int64_t base = 0
) {
    auto const P = static_cast<std::size_t>(total_num_partitions);
    auto const pieces = rapidsmpf::chunk_indices(total_num_rows, P * P);

    // Locally recompute the non-empty (offset, count) sub-regions expected for partition
    // j, in increasing input-region-index (== increasing offset) order.
    std::vector<std::pair<std::int64_t, std::size_t>> expected;
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        auto [start, end] = pieces[static_cast<std::size_t>(i) * P + j];
        if (end > start) {
            expected.emplace_back(base + static_cast<std::int64_t>(start), end - start);
        }
    }

    EXPECT_EQ(received.size(), expected.size());

    // Sort received chunks by their first metadata int64 (== offset) so they align 1:1
    // with the expected list, which is already in offset order.
    std::sort(received.begin(), received.end(), [](auto const& a, auto const& b) {
        std::int64_t oa{}, ob{};
        std::memcpy(&oa, a.metadata->data(), sizeof(std::int64_t));
        std::memcpy(&ob, b.metadata->data(), sizeof(std::int64_t));
        return oa < ob;
    });

    for (std::size_t k = 0; k < received.size() && k < expected.size(); ++k) {
        auto const [off, cnt] = expected[k];
        auto const cs = received[k].stream();
        EXPECT_NO_FATAL_FAILURE(
            validate_packed_data<std::int64_t>(std::move(received[k]), cnt, off, cs, br)
        );
    }
}

/**
 * @brief Execute a full shuffler round-trip and validate every local partition.
 *
 * For each locally-owned partition, generates input data with `make_partition_data`,
 * inserts it into the shuffler, signals insertion completion, then waits (with a
 * 30-second timeout to catch deadlocks) and validates every received partition
 * with `validate_partition_data`.
 *
 * @param shuffler             The shuffler instance under test.
 * @param total_num_partitions Total number of shuffle partitions `P`.
 * @param total_num_rows       Total row count `N` distributed across all sub-regions.
 * @param stream               CUDA stream used for device allocations.
 * @param br                   Buffer resource used to allocate and validate data.
 */
void test_shuffler(
    rapidsmpf::shuffler::Shuffler& shuffler,
    rapidsmpf::shuffler::PartID total_num_partitions,
    std::size_t total_num_rows,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource* br,
    std::int64_t base = 0
) {
    // To expose unexpected deadlocks, we use a 30s timeout. In a normal run, the
    // shuffle shouldn't get near 30s.
    std::chrono::seconds const wait_timeout(30);

    for (rapidsmpf::shuffler::PartID local_pidx : shuffler.local_partitions()) {
        shuffler.insert(make_partition_data(
            total_num_partitions, total_num_rows, local_pidx, stream, *br, base
        ));
    }
    shuffler.insert_finished();

    EXPECT_NO_THROW(shuffler.wait(wait_timeout));

    for (auto local_pidx : shuffler.local_partitions()) {
        validate_partition_data(
            shuffler.extract(local_pidx),
            total_num_partitions,
            total_num_rows,
            local_pidx,
            *br,
            base
        );
    }
}

}  // namespace

class MemoryLimits_NumPartition
    : public ::testing::TestWithParam<
          std::tuple<MemoryLimitsMap, rapidsmpf::shuffler::PartID, std::size_t>> {
  public:
    void SetUp() override {
        stream = rmm::cuda_stream_default;
        std::tie(memory_limits, total_num_partitions, total_num_rows) = GetParam();
        br = rapidsmpf::BufferResource::create(
            rmm::mr::get_current_device_resource_ref(),
            rapidsmpf::PinnedMemoryResource::Disabled,
            memory_limits
        );

        shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
            GlobalEnvironment->comm_,
            0,  // op_id
            total_num_partitions,
            br.get()
        );
    }

    void TearDown() override {
        shuffler.reset();
    }

  protected:
    MemoryLimitsMap memory_limits;
    rapidsmpf::shuffler::PartID total_num_partitions;
    std::size_t total_num_rows;
    rmm::cuda_stream_view stream;
    std::shared_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
};

// test different `memory_available` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
    Shuffler,
    MemoryLimits_NumPartition,
    testing::Combine(
        testing::ValuesIn(
            {get_memory_limits_map(rapidsmpf::MemoryType::HOST),
             get_memory_limits_map(rapidsmpf::MemoryType::DEVICE)}
        ),
        testing::Values(1, 2, 5, 10),  // total_num_partitions
        testing::Values(1, 9, 100, 100'000)  // total_num_rows
    ),
    [](const testing::TestParamInfo<MemoryLimits_NumPartition::ParamType>& info) {
        return std::to_string(info.index) + "__nparts_"
               + std::to_string(std::get<1>(info.param)) + "__nrows_"
               + std::to_string(std::get<2>(info.param));
    }
);

TEST_P(MemoryLimits_NumPartition, round_trip) {
    EXPECT_NO_FATAL_FAILURE(
        test_shuffler(*shuffler, total_num_partitions, total_num_rows, stream, br.get())
    );
}

// Test that the same communicator can be used concurrently by multiple shufflers in
// separate threads
class ConcurrentShuffleTest : public ::testing::TestWithParam<
                                  std::tuple<std::size_t, rapidsmpf::shuffler::PartID>> {
  public:
    void SetUp() override {
        std::tie(num_shufflers, total_num_partitions) = GetParam();

        // these resources will be used by multiple threads to instantiate shufflers
        br =
            rapidsmpf::BufferResource::create(rmm::mr::get_current_device_resource_ref());
        stream = rmm::cuda_stream_default;
    }

    void TearDown() override {}

    // test run for each thread. The test follows the same logic as
    // `MemoryLimits_NumPartition` test, but without any memory limitations
    void RunTest(std::size_t t_id) {
        rapidsmpf::shuffler::Shuffler shuffler(
            GlobalEnvironment->comm_,
            static_cast<rapidsmpf::OpID>(t_id),  // op_id, use t_id as a proxy
            total_num_partitions,
            br.get()
        );

        EXPECT_NO_FATAL_FAILURE(test_shuffler(
            shuffler,
            total_num_partitions,
            100'000,  // total_num_rows
            stream,
            br.get(),
            static_cast<std::int64_t>(t_id)
        ));
    }

    void RunTestTemplate() {
        std::vector<std::future<void>> futures;
        futures.reserve(num_shufflers);

        for (std::size_t t_id = 0; t_id < num_shufflers; t_id++) {
            futures.push_back(std::async(std::launch::async, [this, t_id] {
                ASSERT_NO_FATAL_FAILURE(this->RunTest(t_id));
            }));
        }

        for (auto& f : futures) {
            ASSERT_NO_THROW(f.wait());
        }
    }

    std::size_t num_shufflers;
    rapidsmpf::shuffler::PartID total_num_partitions;

    rmm::cuda_stream_view stream;
    std::shared_ptr<rapidsmpf::BufferResource> br;
};

TEST_P(ConcurrentShuffleTest, round_trip) {
    ASSERT_NO_FATAL_FAILURE(RunTestTemplate());
}

// test different `num_shufflers` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
    ConcurrentShuffle,
    ConcurrentShuffleTest,
    testing::Combine(
        testing::Values(std::size_t{1}, std::size_t{2}, std::size_t{4}),  // num_shufflers
        testing::Values(  // total_num_partitions
            rapidsmpf::shuffler::PartID{1},
            rapidsmpf::shuffler::PartID{10},
            rapidsmpf::shuffler::PartID{100}
        )
    ),
    [](const testing::TestParamInfo<ConcurrentShuffleTest::ParamType>& info) {
        return "num_shufflers_" + std::to_string(std::get<0>(info.param))
               + "__total_num_partitions_" + std::to_string(std::get<1>(info.param));
    }
);

TEST(Shuffler, SpillOnInsertAndExtraction) {
    rapidsmpf::shuffler::PartID const total_num_partitions = 2;
    auto stream = rmm::cuda_stream_default;

    // Use RapidsMPF's memory resource adaptor so the test can observe per-rank
    // allocation counts via `get_main_record().num_current_allocs()`.
    rapidsmpf::RmmResourceAdaptor mr{rmm::mr::get_current_device_resource_ref()};

    // Control spilling by adjusting the DEVICE memory limit at runtime.
    // `memory_available(DEVICE)` is computed as `limit - current_allocated()`, so a
    // sufficiently large positive limit reliably keeps available memory > 0 (no spill),
    // while a sufficiently large negative limit reliably keeps available memory < 0
    // (force spill), regardless of how many bytes are currently allocated from `mr`.
    constexpr std::int64_t k_no_spill_limit = (1LL << 40);
    constexpr std::int64_t k_force_spill_limit = -(1LL << 40);
    auto br = rapidsmpf::BufferResource::create(
        mr,
        rapidsmpf::PinnedMemoryResource::Disabled,
        {{rapidsmpf::MemoryType::DEVICE, k_no_spill_limit}},
        std::nullopt  // disable periodic spill check
    );

    // Create a communicator of size 1, such that each shuffler will run locally.
    auto comm = GlobalEnvironment->split_comm();
    EXPECT_EQ(comm->nranks(), 1);

    // Create a shuffler and input chunks.
    rapidsmpf::shuffler::Shuffler shuffler(
        comm,
        0,  // op_id
        total_num_partitions,
        br.get()
    );
    // Create one non-empty chunk per partition. Each chunk owns a single device
    // buffer, so we start with exactly `total_num_partitions` device allocations.
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> input_chunks;
    for (rapidsmpf::shuffler::PartID pid = 0; pid < total_num_partitions; ++pid) {
        input_chunks.emplace(pid, generate_packed_data(1000, 0, stream, *br));
    }

    // Insert spills does nothing when device memory is available, we start
    // with 2 device allocations.
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);
    shuffler.insert(std::move(input_chunks));
    // And we end with two 2 device allocations.
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

    // Let's force spilling.
    br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_force_spill_limit);

    {
        // Now extract triggers spilling of the partition not being extracted.
        std::vector<rapidsmpf::PackedData> output_chunks = rapidsmpf::unspill_partitions(
            shuffler.extract(0), br.get(), rapidsmpf::AllowOverbooking::YES
        );
        EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);

        // And insert also triggers spilling. We end up with zero device allocations.
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(0, std::move(output_chunks.at(0)));
        shuffler.insert(std::move(chunk));
        EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
    }

    // Extract and unspill both partitions.
    std::vector<rapidsmpf::PackedData> out0 = rapidsmpf::unspill_partitions(
        shuffler.extract(0), br.get(), rapidsmpf::AllowOverbooking::YES
    );
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);
    std::vector<rapidsmpf::PackedData> out1 = rapidsmpf::unspill_partitions(
        shuffler.extract(1), br.get(), rapidsmpf::AllowOverbooking::YES
    );
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

    // Disable spilling and insert the first partition.
    br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_no_spill_limit);
    {
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(0, std::move(out0.at(0)));
        shuffler.insert(std::move(chunk));
    }
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

    // Enable spilling and insert the second partition, which should trigger spilling
    // of both the first partition already in the shuffler and the second partition
    // that are being inserted.
    br->set_memory_limit(rapidsmpf::MemoryType::DEVICE, k_force_spill_limit);
    {
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(1, std::move(out1.at(0)));
        shuffler.insert(std::move(chunk));
    }
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
    shuffler.insert_finished();
}

TEST(FinishCounterTests, zero_local_partitions_immediately_finished) {
    rapidsmpf::shuffler::detail::FinishCounter finish_counter(
        /*nranks=*/2, /*n_local_partitions=*/0
    );

    EXPECT_TRUE(finish_counter.all_finished());
}

TEST(FinishCounterTests, nonzero_local_partitions_finishes_after_all_chunks) {
    rapidsmpf::shuffler::detail::FinishCounter finish_counter(
        /*nranks=*/1, /*n_local_partitions=*/2
    );

    EXPECT_FALSE(finish_counter.all_finished());

    // One rank sends 3 chunks total.
    finish_counter.move_goalpost(0, 3);
    finish_counter.add_finished_chunk();
    finish_counter.add_finished_chunk();
    EXPECT_FALSE(finish_counter.all_finished());

    finish_counter.add_finished_chunk();
    EXPECT_TRUE(finish_counter.all_finished());
}

TEST(FinishCounterTests, multi_rank_completion) {
    auto comm = GlobalEnvironment->comm_;

    if (comm->rank() != 0) {
        GTEST_SKIP() << "Test only runs on rank 0";
    }

    // Use nranks partitions so each rank owns exactly 1 partition (round robin).
    auto out_nparts = rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks());

    auto local_partitions = rapidsmpf::shuffler::Shuffler::local_partitions(
        comm, out_nparts, &rapidsmpf::shuffler::Shuffler::round_robin
    );
    ASSERT_EQ(local_partitions.size(), 1);

    rapidsmpf::shuffler::detail::FinishCounter finish_counter(
        comm->nranks(), local_partitions.size()
    );

    // Not finished yet.
    EXPECT_FALSE(finish_counter.all_finished());

    // For nranks ranks, each rank sends 1 data chunk + 1 control, so
    // move_goalpost(rank, 2) per rank.
    for (rapidsmpf::Rank r = 0; r < comm->nranks(); r++) {
        finish_counter.move_goalpost(r, 2);
    }

    // Add finished chunks: 1 data chunk per rank + 1 control per rank = 2 * nranks
    for (rapidsmpf::Rank r = 0; r < comm->nranks(); r++) {
        finish_counter.add_finished_chunk();  // data chunk
        finish_counter.add_finished_chunk();  // control chunk
    }

    EXPECT_TRUE(finish_counter.all_finished());
}

class FinishCounterMultithreadingTest
    : public ::testing::TestWithParam<
          std::tuple<rapidsmpf::shuffler::PartID, std::uint32_t>> {
  protected:
    rapidsmpf::Rank const nranks{1};  // simulate a single rank

    std::unique_ptr<rapidsmpf::shuffler::detail::FinishCounter> finish_counter;
    rapidsmpf::shuffler::PartID npartitions;
    std::uint32_t nthreads;

    void SetUp() override {
        std::tie(npartitions, nthreads) = GetParam();

        finish_counter = std::make_unique<rapidsmpf::shuffler::detail::FinishCounter>(
            nranks, npartitions
        );
    }

    void produce_data() {
        // Simulate nranks=1: one rank reports chunk count = npartitions + 1
        // (one data chunk per partition + 1 control message)
        finish_counter->move_goalpost(rapidsmpf::Rank{0}, npartitions + 1);
        for (rapidsmpf::shuffler::PartID i = 0; i <= npartitions; i++) {
            finish_counter->add_finished_chunk();
        }
    }
};

// Parametrize on number of partitions and number of consumer threads
INSTANTIATE_TEST_SUITE_P(
    FinishCounterMultithreadingTestP,
    FinishCounterMultithreadingTest,
    testing::Combine(testing::Values(1, 2, 100, 101), testing::Values(1, 2, 3)),
    [](const auto& info) {
        return "npartitions_" + std::to_string(std::get<0>(info.param)) + "__nthreads_"
               + std::to_string(std::get<1>(info.param));
    }
);

TEST_P(FinishCounterMultithreadingTest, concurrent_all_finished_check) {
    produce_data();

    std::atomic<std::uint32_t> n_checks{0};
    std::vector<std::future<void>> futures;
    for (std::uint32_t tid = 0; tid < nthreads; tid++) {
        futures.emplace_back(std::async(std::launch::async, [&, tid] {
            for (std::uint32_t i = tid; i < npartitions; i += nthreads) {
                EXPECT_TRUE(finish_counter->all_finished());
                n_checks.fetch_add(1, std::memory_order_relaxed);
            }
        }));
    }

    EXPECT_NO_THROW(std::ranges::for_each(futures, [](auto& f) { f.get(); }));

    EXPECT_EQ(npartitions, n_checks);
    EXPECT_TRUE(finish_counter->all_finished());
}

class ContiguousPartitionAssignmentTest
    : public ::testing::TestWithParam<rapidsmpf::shuffler::PartID> {
  protected:
    void SetUp() override {
        comm = GlobalEnvironment->comm_;
        nranks = comm->nranks();
        rank = comm->rank();
        total_num_partitions = GetParam();
    }

    std::shared_ptr<rapidsmpf::Communicator> comm;
    rapidsmpf::Rank nranks;
    rapidsmpf::Rank rank;
    rapidsmpf::shuffler::PartID total_num_partitions;
};

INSTANTIATE_TEST_SUITE_P(
    PartitionAssignment,
    ContiguousPartitionAssignmentTest,
    testing::Values(1, 2, 3, 5, 7, 10, 16, 100),
    [](const testing::TestParamInfo<rapidsmpf::shuffler::PartID>& info) {
        return "nparts_" + std::to_string(info.param);
    }
);

TEST_P(ContiguousPartitionAssignmentTest, contiguous) {
    std::vector<std::vector<rapidsmpf::shuffler::PartID>> rank_partitions(nranks);
    for (rapidsmpf::shuffler::PartID pid = 0; pid < total_num_partitions; ++pid) {
        auto owner =
            rapidsmpf::shuffler::Shuffler::contiguous(comm, pid, total_num_partitions);
        EXPECT_GE(owner, 0);
        EXPECT_LT(owner, nranks);
        rank_partitions[owner].push_back(pid);
    }

    // Each rank's partitions must be contiguous.
    for (rapidsmpf::Rank r = 0; r < nranks; ++r) {
        auto const& pids = rank_partitions[r];
        for (std::size_t i = 1; i < pids.size(); ++i) {
            EXPECT_EQ(pids[i], pids[i - 1] + 1);
        }
    }

    // Concatenating all rank partitions should cover [0, total_num_partitions).
    std::vector<rapidsmpf::shuffler::PartID> all_pids;
    for (auto const& pids : rank_partitions) {
        all_pids.insert(all_pids.end(), pids.begin(), pids.end());
    }
    EXPECT_EQ(all_pids, iota_vector<rapidsmpf::shuffler::PartID>(total_num_partitions));
}

TEST(Shuffler, ShutdownWhilePaused) {
    auto progress_thread = GlobalEnvironment->comm_->progress_thread();
    auto mr = rmm::mr::get_current_device_resource_ref();

    auto br = rapidsmpf::BufferResource::create(mr);

    auto shuffler =
        rapidsmpf::shuffler::Shuffler(GlobalEnvironment->comm_, 0, 1, br.get());

    progress_thread->pause();
    EXPECT_FALSE(progress_thread->is_running());
    shuffler.insert_finished();
    // Progress thread must be running before shuffle shutdown, otherwise we have some
    // orphan messages in the shuffle that are never sent/received.
    progress_thread->resume();
    EXPECT_TRUE(progress_thread->is_running());
    EXPECT_NO_THROW(shuffler.shutdown());
}

class ExtractEmptyPartitionsTest : public ::testing::Test {
  public:
    static constexpr rapidsmpf::shuffler::PartID nparts = 10;
    static constexpr auto wait_timeout = std::chrono::seconds(30);

    void SetUp() override {
        stream = rmm::cuda_stream_default;
        br =
            rapidsmpf::BufferResource::create(rmm::mr::get_current_device_resource_ref());

        shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
            GlobalEnvironment->comm_, 0, nparts, br.get()
        );
    }

    void TearDown() override {
        shuffler.reset();
    }

    void insert_chunks(
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>&& chunks
    ) {
        if (!chunks.empty()) {
            shuffler->insert(std::move(chunks));
        }
        shuffler->insert_finished();
    }

    void verify_extracted_chunks(auto expected_empty_fn) {
        EXPECT_NO_THROW(shuffler->wait(wait_timeout));
        for (auto pid : shuffler->local_partitions()) {
            SCOPED_TRACE("pid: " + std::to_string(pid));
            std::vector<rapidsmpf::PackedData> chunks;
            EXPECT_NO_THROW({ chunks = shuffler->extract(pid); });

            if (expected_empty_fn(pid)) {
                EXPECT_TRUE(chunks.empty());
            } else {
                EXPECT_EQ(GlobalEnvironment->comm_->nranks(), chunks.size());
            }
        }
    }

    auto empty_packed_data() {
        return rapidsmpf::PackedData{
            std::make_unique<std::vector<std::uint8_t>>(),
            br->move(std::make_unique<rmm::device_buffer>(), stream)
        };
    }

    auto non_empty_packed_data() {
        return rapidsmpf::PackedData{
            std::make_unique<std::vector<std::uint8_t>>(10),
            br->move(std::make_unique<rmm::device_buffer>(10, stream), stream)
        };
    }

    rmm::cuda_stream_view stream;
    std::shared_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
};

TEST_F(ExtractEmptyPartitionsTest, NoInsertions) {
    insert_chunks({});
    EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto) { return true; }));
}

TEST_F(ExtractEmptyPartitionsTest, AllEmptyInsertions) {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
    for (rapidsmpf::shuffler::PartID pid = 0; pid < nparts; ++pid) {
        chunks.emplace(pid, empty_packed_data());
    }

    insert_chunks(std::move(chunks));
    EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto) { return true; }));
}

TEST_F(ExtractEmptyPartitionsTest, SomeEmptyInsertions) {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
    for (rapidsmpf::shuffler::PartID pid = 0; pid < nparts; ++pid) {
        if (pid % 3 == 0) {
            chunks.emplace(pid, empty_packed_data());
        }
    }

    insert_chunks(std::move(chunks));
    EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto) { return true; }));
}

TEST_F(ExtractEmptyPartitionsTest, SomeEmptyAndNonEmptyInsertions) {
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;
    for (rapidsmpf::shuffler::PartID pid = 0; pid < nparts; ++pid) {
        if (pid % 3 == 0) {
            chunks.emplace(pid, empty_packed_data());
        } else {
            chunks.emplace(pid, non_empty_packed_data());
        }
    }

    insert_chunks(std::move(chunks));
    EXPECT_NO_FATAL_FAILURE(verify_extracted_chunks([](auto pid) {
        return pid % 3 == 0;
    }));
}

TEST(ShufflerTest, multiple_shutdowns) {
    auto& comm = GlobalEnvironment->comm_;
    auto br =
        rapidsmpf::BufferResource::create(rmm::mr::get_current_device_resource_ref());
    auto shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
        comm, 0, comm->nranks(), br.get()
    );

    shuffler->insert_finished();
    EXPECT_NO_THROW(shuffler->wait(std::chrono::seconds(30)));
    for (auto pid : shuffler->local_partitions()) {
        std::ignore = shuffler->extract(pid);
    }

    constexpr int n_threads = 10;
    std::vector<std::future<void>> futures;
    for (int i = 0; i < n_threads; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&] {
            shuffler->shutdown();
        }));
    }
    std::ranges::for_each(futures, [](auto& future) { future.get(); });
}

// Test that multiple threads can call wait() concurrently.
TEST(Shuffler, concurrent_wait) {
    auto const& comm = GlobalEnvironment->comm_;
    auto br =
        rapidsmpf::BufferResource::create(rmm::mr::get_current_device_resource_ref());

    // Use more partitions than ranks so each rank owns multiple partitions, ensuring
    // multiple threads call wait() concurrently on the same shuffler.
    auto const total_num_partitions =
        rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks()) * 8;
    constexpr std::size_t total_num_rows = 1000;
    constexpr auto wait_timeout = std::chrono::seconds{30};

    rapidsmpf::shuffler::Shuffler shuffler(comm, 0, total_num_partitions, br.get());

    // Insert each owned input region concurrently, each thread using its own pool stream.
    {
        std::vector<std::future<void>> insert_futures;
        for (rapidsmpf::shuffler::PartID local_pidx : shuffler.local_partitions()) {
            insert_futures.push_back(std::async(std::launch::async, [&, local_pidx] {
                shuffler.insert(make_partition_data(
                    total_num_partitions,
                    total_num_rows,
                    local_pidx,
                    br->stream_pool().get_stream(),
                    *br
                ));
            }));
        }
        std::ranges::for_each(insert_futures, [](auto& f) { f.get(); });
        shuffler.insert_finished();
    }

    // Wait + extract + validate each local partition concurrently, so multiple threads
    // call wait() on the same shuffler at once.
    std::vector<std::future<void>> futures;
    for (auto j : shuffler.local_partitions()) {
        futures.push_back(std::async(std::launch::async, [&, j] {
            EXPECT_NO_THROW(shuffler.wait(wait_timeout));
            validate_partition_data(
                shuffler.extract(j), total_num_partitions, total_num_rows, j, *br
            );
        }));
    }
    std::ranges::for_each(futures, [](auto& f) { f.get(); });
}

// Test that reusing an OpID after a completed shuffle doesn't cause cross-matching of
// messages between the old and new shuffle.
//
// On rank 0 we inject a stream-ordered delay into device allocations so that received
// chunks stay "not ready" in the event loop. With small messages, other ranks can finish
// and move on to the next shuffle. Its messages will then be matched on rank 0 by the
// blocked previous shuffle, unless recv gating correctly prevents cross-talk.
TEST(Shuffler, opid_reuse) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() == 1) {
        GTEST_SKIP() << "OpID reuse test requires multiple ranks";
    }

    auto stream = rmm::cuda_stream_default;
    auto const total_num_partitions =
        rapidsmpf::safe_cast<rapidsmpf::shuffler::PartID>(comm->nranks());
    constexpr std::size_t total_num_rows = 1000;
    constexpr rapidsmpf::OpID op_id = 0;
    constexpr auto wait_timeout = std::chrono::seconds{30};

    rmm::mr::cuda_memory_resource mr;
    auto br = rapidsmpf::BufferResource::create(mr);

    // On rank 0, wrap the device MR with a delayed version for the shuffler.
    std::unique_ptr<DelayedMemoryResource> delayed_mr;
    std::shared_ptr<rapidsmpf::BufferResource> delayed_br;
    rapidsmpf::BufferResource* shuffler_br = br.get();
    if (comm->rank() == 0) {
        delayed_mr =
            std::make_unique<DelayedMemoryResource>(mr, std::chrono::milliseconds(500));
        delayed_br = rapidsmpf::BufferResource::create(*delayed_mr);
        shuffler_br = delayed_br.get();
    }

    // Each shuffle uses a distinct base offset (in place of a seed) so the two shuffles
    // carry different data; a cross-matched message would therefore fail validation.
    auto insert_data = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t base) {
        for (rapidsmpf::shuffler::PartID local_pidx : shuffler.local_partitions()) {
            shuffler.insert(make_partition_data(
                total_num_partitions, total_num_rows, local_pidx, stream, *br, base
            ));
        }
    };

    auto validate_results = [&](rapidsmpf::shuffler::Shuffler& shuffler,
                                std::int64_t base) {
        for (auto j : shuffler.local_partitions()) {
            validate_partition_data(
                shuffler.extract(j), total_num_partitions, total_num_rows, j, *br, base
            );
        }
    };

    rapidsmpf::shuffler::Shuffler shuffle1(
        comm, op_id, total_num_partitions, shuffler_br
    );
    insert_data(shuffle1, 42);
    shuffle1.insert_finished();
    EXPECT_NO_THROW(shuffle1.wait(wait_timeout));

    rapidsmpf::shuffler::Shuffler shuffle2(
        comm, op_id, total_num_partitions, shuffler_br
    );
    insert_data(shuffle2, 123);
    shuffle2.insert_finished();
    EXPECT_NO_THROW(shuffle2.wait(wait_timeout));

    validate_results(shuffle1, 42);
    validate_results(shuffle2, 123);
}

// Same as opid_reuse but with total_num_partitions=1, so only rank 0 owns a partition.
// All other ranks have empty local_partitions and empty recv loops. This exercises the
// edge case where non-partition-owning ranks must still correctly handle op_id reuse.
TEST(Shuffler, opid_reuse_with_empty_partitions) {
    auto const& comm = GlobalEnvironment->comm_;
    if (comm->nranks() == 1) {
        GTEST_SKIP() << "OpID reuse test requires multiple ranks";
    }

    auto stream = rmm::cuda_stream_default;
    constexpr rapidsmpf::shuffler::PartID total_num_partitions = 1;
    constexpr std::size_t total_num_rows = 1000;
    constexpr rapidsmpf::OpID op_id = 0;
    constexpr auto wait_timeout = std::chrono::seconds{30};

    rmm::mr::cuda_memory_resource mr;
    auto br = rapidsmpf::BufferResource::create(mr);

    // On rank 0, wrap the device MR with a delayed version for the shuffler.
    std::unique_ptr<DelayedMemoryResource> delayed_mr;
    std::shared_ptr<rapidsmpf::BufferResource> delayed_br;
    rapidsmpf::BufferResource* shuffler_br = br.get();
    if (comm->rank() == 0) {
        delayed_mr =
            std::make_unique<DelayedMemoryResource>(mr, std::chrono::milliseconds(500));
        delayed_br = rapidsmpf::BufferResource::create(*delayed_mr);
        shuffler_br = delayed_br.get();
    }

    // Each shuffle uses a distinct base offset (in place of a seed) so the two shuffles
    // carry different data; a cross-matched message would therefore fail validation.
    // With total_num_partitions=1, only rank 0 owns the single partition; all other ranks
    // have empty local_partitions(), so they insert/validate nothing.
    auto insert_data = [&](rapidsmpf::shuffler::Shuffler& shuffler, std::int64_t base) {
        for (rapidsmpf::shuffler::PartID local_pidx : shuffler.local_partitions()) {
            shuffler.insert(make_partition_data(
                total_num_partitions, total_num_rows, local_pidx, stream, *br, base
            ));
        }
    };

    auto validate_results = [&](rapidsmpf::shuffler::Shuffler& shuffler,
                                std::int64_t base) {
        for (auto j : shuffler.local_partitions()) {
            validate_partition_data(
                shuffler.extract(j), total_num_partitions, total_num_rows, j, *br, base
            );
        }
    };

    rapidsmpf::shuffler::Shuffler shuffle1(
        comm, op_id, total_num_partitions, shuffler_br
    );
    insert_data(shuffle1, 42);
    shuffle1.insert_finished();
    EXPECT_NO_THROW(shuffle1.wait(wait_timeout));

    rapidsmpf::shuffler::Shuffler shuffle2(
        comm, op_id, total_num_partitions, shuffler_br
    );
    insert_data(shuffle2, 123);
    shuffle2.insert_finished();
    EXPECT_NO_THROW(shuffle2.wait(wait_timeout));

    validate_results(shuffle1, 42);
    validate_results(shuffle2, 123);
}
