/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <future>
#include <memory>

#include <gtest/gtest.h>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/buffer/resource.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils.hpp>

#include "environment.hpp"
#include "utils.hpp"

extern Environment* GlobalEnvironment;

class NumOfPartitions : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {};

// test different `num_partitions` and `num_rows`.
INSTANTIATE_TEST_SUITE_P(
    Shuffler,
    NumOfPartitions,
    testing::Combine(
        testing::Range(1, 10),  // num_partitions
        testing::Range(1, 100, 9)  // num_rows
    )
);

TEST_P(NumOfPartitions, partition_and_pack) {
    int const num_partitions = std::get<0>(GetParam());
    int const num_rows = std::get<1>(GetParam());
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    cudf::table expect =
        random_table_with_index(seed, static_cast<std::size_t>(num_rows), 0, 10);

    auto chunks = rapidsmpf::partition_and_pack(
        expect, {1}, num_partitions, hash_fn, seed, stream, mr
    );

    // Convert to a vector
    std::vector<rapidsmpf::PackedData> chunks_vector;
    for (auto& [_, chunk] : chunks) {
        chunks_vector.push_back(std::move(chunk));
    }
    EXPECT_EQ(chunks_vector.size(), num_partitions);

    auto result = rapidsmpf::unpack_and_concat(std::move(chunks_vector), stream, mr);

    // Compare the input table with the result. We ignore the row order by
    // sorting by their index (first column).
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
}

TEST_P(NumOfPartitions, split_and_pack) {
    int const num_partitions = std::get<0>(GetParam());
    int const num_rows = std::get<1>(GetParam());
    std::int64_t const seed = 42;
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    cudf::table expect = random_table_with_index(seed, num_rows, 0, 10);

    std::vector<cudf::size_type> splits;
    for (int i = 1; i < num_partitions; ++i) {
        splits.emplace_back(i * num_rows / num_partitions);
    }

    auto chunks = rapidsmpf::split_and_pack(expect, splits, stream, mr);

    // Convert to a vector (restoring the original order).
    std::vector<rapidsmpf::PackedData> chunks_vector;
    for (int i = 0; i < num_partitions; ++i) {
        chunks_vector.emplace_back(std::move(chunks.at(i)));
    }
    EXPECT_EQ(chunks_vector.size(), num_partitions);

    auto result = rapidsmpf::unpack_and_concat(std::move(chunks_vector), stream, mr);

    // Compare the input table with the result.
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect, *result);
}

TEST(MetadataMessage, round_trip) {
    auto stream = cudf::get_default_stream();

    auto metadata = iota_vector<uint8_t>(100);

    auto expect = rapidsmpf::shuffler::detail::Chunk::from_packed_data(
        1,  // chunk_id
        2,  // part_id
        {std::make_unique<std::vector<uint8_t>>(metadata), nullptr},  // packed_data
        nullptr,  // event
        stream,
        nullptr  // buffer_resource
    );

    // Extract the metadata from then chunk.
    auto msg = expect.serialize();
    EXPECT_FALSE(expect.is_metadata_buffer_set());

    // Create a new chunk by deserializing the message.
    auto result = rapidsmpf::shuffler::detail::Chunk::deserialize(*msg);

    // They should be identical.
    EXPECT_EQ(expect.part_id(0), result.part_id(0));
    EXPECT_EQ(expect.chunk_id(), result.chunk_id());
    EXPECT_EQ(expect.expected_num_chunks(0), result.expected_num_chunks(0));
    EXPECT_EQ(expect.concat_data_size(), result.concat_data_size());
    EXPECT_EQ(expect.concat_metadata_size(), result.concat_metadata_size());

    // The metadata should be identical to the original.
    EXPECT_EQ(metadata, *result.release_metadata_buffer());
}

using MemoryAvailableMap =
    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>;

// Help function to get the `memory_available` argument for a `BufferResource`
// that prioritizes the specified memory type.
MemoryAvailableMap get_memory_available_map(rapidsmpf::MemoryType priorities) {
    using namespace rapidsmpf;

    // We set all memory types to use an available function that always return zero.
    BufferResource::MemoryAvailable always_zero = []() -> std::int64_t { return 0; };
    MemoryAvailableMap ret = {
        {MemoryType::DEVICE, always_zero}, {MemoryType::HOST, always_zero}
    };
    // And then set the prioritized memory type to use the max function.
    ret.at(priorities) = std::numeric_limits<std::int64_t>::max;
    return ret;
}

/// @tparam InsertFn: lambda that inserts the packed chunks into the shuffler.
/// Signature: void(std::vector<rapidsmpf::PackedData>&& packed_chunks)
/// @tparam InsertFinishedFn: lambda that inserts the finished flag into the shuffler.
/// Signature: void()
template <typename InsertFn, typename InsertFinishedFn>
void test_shuffler(
    std::shared_ptr<rapidsmpf::Communicator> const& comm,
    rapidsmpf::shuffler::Shuffler& shuffler,
    rapidsmpf::shuffler::PartID total_num_partitions,
    InsertFn&& insert_fn,
    InsertFinishedFn&& insert_finished_fn,
    std::size_t total_num_rows,
    std::int64_t seed,
    cudf::hash_id hash_fn,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    // To expose unexpected deadlocks, we use a 30s timeout. In a normal run, the shuffle
    // shouldn't get near 30s.
    std::chrono::milliseconds const wait_timeout(30 * 1000);

    // Every rank creates the full input table and all the expected partitions (also
    // partitions this rank might not get after the shuffle).
    cudf::table full_input_table = random_table_with_index(seed, total_num_rows, 0, 10);
    auto [expect_partitions, owner] = rapidsmpf::partition_and_split(
        full_input_table,
        {1},
        static_cast<std::int32_t>(total_num_partitions),
        hash_fn,
        seed,
        stream,
        mr
    );

    cudf::size_type row_offset = 0;
    cudf::size_type partiton_size =
        full_input_table.num_rows() / static_cast<cudf::size_type>(total_num_partitions);
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        // To simulate that `full_input_table` is distributed between multiple ranks,
        // we divided them into `total_num_partitions` number of partitions and pick
        // the partitions this rank should use as input. We pick using round robin but
        // any distribution would work (as long as no rows are picked by multiple ranks).
        // TODO: we should test different distributions of the input partitions.
        if (rapidsmpf::shuffler::Shuffler::round_robin(comm, i) == comm->rank()) {
            cudf::size_type row_end = row_offset + partiton_size;
            if (i == total_num_partitions - 1) {
                // Include the reminder of rows in the very last partition.
                row_end = full_input_table.num_rows();
            }
            // Select the partition from the full input table.
            auto slice = cudf::slice(full_input_table, {row_offset, row_end}).at(0);
            // Hash the `slice` into chunks and pack (serialize) them.
            auto packed_chunks = rapidsmpf::partition_and_pack(
                slice,
                {1},
                static_cast<std::int32_t>(total_num_partitions),
                hash_fn,
                seed,
                stream,
                mr
            );
            // Add the chunks to the shuffle
            insert_fn(std::move(packed_chunks));
        }
        row_offset += partiton_size;
    }
    // Tell the shuffler that we have no more input partitions.
    insert_finished_fn();

    while (!shuffler.finished()) {
        auto finished_partition = shuffler.wait_any(wait_timeout);
        auto packed_chunks = shuffler.extract(finished_partition);
        auto result = rapidsmpf::unpack_and_concat(std::move(packed_chunks), stream, mr);

        // We should only receive the partitions assigned to this rank.
        EXPECT_EQ(shuffler.partition_owner(comm, finished_partition), comm->rank());

        // Check the result while ignoring the row order.
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(
            sort_table(result), sort_table(expect_partitions[finished_partition])
        );
    }

    shuffler.shutdown();
}

class MemoryAvailable_NumPartition
    : public cudf::test::BaseFixtureWithParam<
          std::tuple<MemoryAvailableMap, rapidsmpf::shuffler::PartID, std::size_t>> {
  public:
    void SetUp() override {
        stream = cudf::get_default_stream();
        memory_available = std::get<0>(GetParam());
        total_num_partitions = std::get<1>(GetParam());
        total_num_rows = std::get<2>(GetParam());
        br = std::make_unique<rapidsmpf::BufferResource>(mr(), memory_available);

        shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            0,  // op_id
            total_num_partitions,
            stream,
            br.get()
        );

        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

  protected:
    MemoryAvailableMap memory_available;
    rapidsmpf::shuffler::PartID total_num_partitions;
    std::size_t total_num_rows;
    std::int64_t seed = 42;
    cudf::hash_id hash_fn = cudf::hash_id::HASH_MURMUR3;
    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
};

// test different `memory_available` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
    Shuffler,
    MemoryAvailable_NumPartition,
    testing::Combine(
        testing::ValuesIn(
            {get_memory_available_map(rapidsmpf::MemoryType::HOST),
             get_memory_available_map(rapidsmpf::MemoryType::DEVICE)}
        ),
        testing::Values(1, 2, 5, 10),  // total_num_partitions
        testing::Values(1, 9, 100)  // total_num_rows
    )
);

// both insert and insert_finished ungrouped
TEST_P(MemoryAvailable_NumPartition, round_trip) {
    EXPECT_NO_FATAL_FAILURE(test_shuffler(
        GlobalEnvironment->comm_,
        *shuffler,
        total_num_partitions,
        [&](auto&& packed_chunks) { shuffler->insert(std::move(packed_chunks)); },
        [&]() {
            for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
                shuffler->insert_finished(i);
            }
        },
        total_num_rows,
        seed,
        hash_fn,
        stream,
        mr()
    ));
}

// both insert and insert_finished grouped
TEST_P(MemoryAvailable_NumPartition, round_trip_both_grouped) {
    EXPECT_NO_FATAL_FAILURE(test_shuffler(
        GlobalEnvironment->comm_,
        *shuffler,
        total_num_partitions,
        [&](auto&& packed_chunks) { shuffler->insert_grouped(std::move(packed_chunks)); },
        [&]() {
            shuffler->insert_finished(
                iota_vector<rapidsmpf::shuffler::PartID>(total_num_partitions)
            );
        },
        total_num_rows,
        seed,
        hash_fn,
        stream,
        mr()
    ));
}

// insert grouped and insert_finished ungrouped
TEST_P(MemoryAvailable_NumPartition, round_trip_insert_grouped) {
    EXPECT_NO_FATAL_FAILURE(test_shuffler(
        GlobalEnvironment->comm_,
        *shuffler,
        total_num_partitions,
        [&](auto&& packed_chunks) { shuffler->insert_grouped(std::move(packed_chunks)); },
        [&]() {
            for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
                shuffler->insert_finished(i);
            }
        },
        total_num_rows,
        seed,
        hash_fn,
        stream,
        mr()
    ));
}

// insert ungrouped and insert_finished grouped
TEST_P(MemoryAvailable_NumPartition, round_trip_finished_grouped) {
    EXPECT_NO_FATAL_FAILURE(test_shuffler(
        GlobalEnvironment->comm_,
        *shuffler,
        total_num_partitions,
        [&](auto&& packed_chunks) { shuffler->insert(std::move(packed_chunks)); },
        [&]() {
            shuffler->insert_finished(
                iota_vector<rapidsmpf::shuffler::PartID>(total_num_partitions)
            );
        },
        total_num_rows,
        seed,
        hash_fn,
        stream,
        mr()
    ));
}

// Test that the same communicator can be used concurrently by multiple shufflers in
// separate threads
class ConcurrentShuffleTest
    : public cudf::test::BaseFixtureWithParam<std::tuple<int, int>> {
  public:
    void SetUp() override {
        num_shufflers = std::get<0>(GetParam());
        total_num_partitions =
            static_cast<rapidsmpf::shuffler::PartID>(std::get<1>(GetParam()));

        // these resources will be used by multiple threads to instantiate shufflers
        br = std::make_unique<rapidsmpf::BufferResource>(mr());
        stream = cudf::get_default_stream();

        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

    // test run for each thread. The test follows the same logic as
    // `MemoryAvailable_NumPartition` test, but without any memory limitations
    template <typename InsertFn, typename InsertFinishedFn>
    void RunTest(int t_id, InsertFn&& insert_fn, InsertFinishedFn&& insert_finished_fn) {
        rapidsmpf::shuffler::Shuffler shuffler(
            GlobalEnvironment->comm_,
            GlobalEnvironment->progress_thread_,
            t_id,  // op_id, use t_id as a proxy
            total_num_partitions,
            stream,
            br.get()
        );

        EXPECT_NO_FATAL_FAILURE(test_shuffler(
            GlobalEnvironment->comm_,
            shuffler,
            total_num_partitions,
            [&](auto&& packed_chunks) { insert_fn(shuffler, std::move(packed_chunks)); },
            [&]() { insert_finished_fn(shuffler); },
            100,  // total_num_rows
            t_id,  // seed
            cudf::hash_id::HASH_MURMUR3,
            stream,
            mr()
        ));
    }

    template <typename InsertFn, typename InsertFinishedFn>
    void RunTestTemplate(InsertFn insert_fn, InsertFinishedFn insert_finished_fn) {
        std::vector<std::future<void>> futures;
        futures.reserve(static_cast<std::size_t>(num_shufflers));

        for (int t_id = 0; t_id < num_shufflers; t_id++) {
            // pass a copy of the insert_fn and insert_finished_fn to each thread
            futures.push_back(std::async(
                std::launch::async,
                [this,
                 t_id,
                 insert_fn1 = insert_fn,
                 insert_finished_fn1 = insert_finished_fn] {
                    ASSERT_NO_FATAL_FAILURE(this->RunTest(
                        t_id, std::move(insert_fn1), std::move(insert_finished_fn1)
                    ));
                }
            ));
        }

        for (auto& f : futures) {
            ASSERT_NO_THROW(f.wait());
        }
    }

    int num_shufflers;
    rapidsmpf::shuffler::PartID total_num_partitions;

    rmm::cuda_stream_view stream;
    std::unique_ptr<rapidsmpf::BufferResource> br;
};

// both insert and insert_finished ungrouped
TEST_P(ConcurrentShuffleTest, round_trip) {
    ASSERT_NO_FATAL_FAILURE(RunTestTemplate(
        [&](auto& shuffler, auto&& packed_chunks) {
            shuffler.insert(std::move(packed_chunks));
        },
        [&](auto& shuffler) {
            for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
                shuffler.insert_finished(i);
            }
        }
    ));
}

// both insert and insert_finished grouped
TEST_P(ConcurrentShuffleTest, round_trip_both_grouped) {
    ASSERT_NO_FATAL_FAILURE(RunTestTemplate(
        [&](auto& shuffler, auto&& packed_chunks) {
            shuffler.insert_grouped(std::move(packed_chunks));
        },
        [&](auto& shuffler) {
            shuffler.insert_finished(
                iota_vector<rapidsmpf::shuffler::PartID>(total_num_partitions)
            );
        }
    ));
}

// insert grouped and insert_finished ungrouped
TEST_P(ConcurrentShuffleTest, round_trip_insert_grouped) {
    ASSERT_NO_FATAL_FAILURE(RunTestTemplate(
        [&](auto& shuffler, auto&& packed_chunks) {
            shuffler.insert_grouped(std::move(packed_chunks));
        },
        [&](auto& shuffler) {
            for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
                shuffler.insert_finished(i);
            }
        }
    ));
}

// insert ungrouped and insert_finished grouped
TEST_P(ConcurrentShuffleTest, round_trip_finished_grouped) {
    ASSERT_NO_FATAL_FAILURE(RunTestTemplate(
        [&](auto& shuffler, auto&& packed_chunks) {
            shuffler.insert(std::move(packed_chunks));
        },
        [&](auto& shuffler) {
            shuffler.insert_finished(
                iota_vector<rapidsmpf::shuffler::PartID>(total_num_partitions)
            );
        }
    ));
}

// test different `num_shufflers` and `total_num_partitions`.
INSTANTIATE_TEST_SUITE_P(
    ConcurrentShuffle,
    ConcurrentShuffleTest,
    testing::Combine(
        testing::ValuesIn({1, 2, 4}),  // num_shufflers
        testing::ValuesIn({1, 10, 100})  // total_num_partitions
    ),
    [](const testing::TestParamInfo<ConcurrentShuffleTest::ParamType>& info) {
        return "num_shufflers_" + std::to_string(std::get<0>(info.param))
               + "__total_num_partitions_" + std::to_string(std::get<1>(info.param));
    }
);

/// test case for `insert_grouped` and `insert_finished`. This test would only test the
/// insertion logic, so, the progress thread is paused for the duration of the test. This
/// will prevent the progress thread from extracting from the outgoing_postbox_. Also,
/// we disable periodic spill check to avoid the buffer resource from spilling chunks in
/// the ready postbox.
class ShuffleInsertGroupedTest
    : public cudf::test::BaseFixtureWithParam<std::tuple<size_t, size_t>> {
  public:
    void SetUp() override {
        pids = iota_vector<rapidsmpf::shuffler::PartID>(std::get<0>(GetParam()));

        num_bytes = std::get<1>(GetParam());

        stream = cudf::get_default_stream();

        progress_thread =
            std::make_shared<rapidsmpf::ProgressThread>(GlobalEnvironment->comm_->logger()
            );

        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        // resume progress thread - this will guarantee that shuffler progress function is
        // marked as done. This is important to ensure that the test does not hang.
        progress_thread->resume();

        if (shuffler) {
            shuffler->shutdown();
        }

        progress_thread->stop();
        GlobalEnvironment->barrier();
    }

    // Helper method to generate packed data for testing
    std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData>
    generate_packed_data() {
        auto dummy_meta = std::make_unique<std::vector<std::uint8_t>>(num_bytes);
        // this will wrap around after 256 bytes
        std::iota(dummy_meta->begin(), dummy_meta->end(), 0);

        // use the same buffer for all partitions
        auto dummy_data =
            std::make_unique<rmm::device_buffer>(dummy_meta->data(), num_bytes, stream);

        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunks;

        for (auto pid : pids) {
            chunks[pid] = rapidsmpf::PackedData(
                std::make_unique<std::vector<std::uint8_t>>(*dummy_meta),
                std::make_unique<rmm::device_buffer>(
                    dummy_data->data(), num_bytes, stream
                )
            );
        }
        cudaStreamSynchronize(stream);

        return chunks;
    }

    // Helper method to verify the shuffler state after insert
    void verify_shuffler_state(rapidsmpf::shuffler::Shuffler& shuffler) {
        rapidsmpf::Rank local_rank = GlobalEnvironment->comm_->rank(),
                        n_ranks = GlobalEnvironment->comm_->nranks();

        // Rank -> n_messages in a chunk
        std::vector<int64_t> expected_n_messages(n_ranks, 0);
        for (auto pid : pids) {
            rapidsmpf::Rank target =
                shuffler.partition_owner(GlobalEnvironment->comm_, pid);
            expected_n_messages[target]++;
        }

        // recreate the outbound_chunks map from the shuffler state
        std::unordered_map<
            rapidsmpf::shuffler::PartID,
            rapidsmpf::shuffler::detail::ChunkID>
            outbound_chunks;

        size_t n_control_messages = 0;
        size_t n_data_messages = 0;
        size_t n_total_data_size = 0;
        size_t n_total_metadata_size = 0;

        // outgoing postbox contains messages to all remote ranks (including finished
        // partitions message) its key is the rank ID.
        for (rapidsmpf::Rank rank = 0; rank < n_ranks; ++rank) {
            if (rank == local_rank || expected_n_messages[rank] == 0)
            {  // skip local rank
                continue;
            }

            auto chunks = shuffler.outgoing_postbox_.extract_by_key(rank);
            for (auto& [cid, chunk] : chunks) {
                for (size_t i = 0; i < chunk.n_messages(); ++i) {
                    outbound_chunks[chunk.part_id(i)]++;
                    if (chunk.is_control_message(i)) {
                        n_control_messages++;
                    } else {
                        n_data_messages++;
                        n_total_data_size += chunk.data_size(i);
                        n_total_metadata_size += chunk.metadata_size(i);
                    }
                }
            }
        }
        EXPECT_TRUE(shuffler.outgoing_postbox_.empty());

        // ready postbox contains all local messages. Its key is the partition ID.
        for (auto pid : pids) {
            // only check for local partitions
            if (shuffler.partition_owner(GlobalEnvironment->comm_, pid) == local_rank) {
                auto local_chunks = shuffler.ready_postbox_.extract_by_key(pid);
                // manually add the control message as ready postbox does not contain
                // control messages
                outbound_chunks[pid]++;
                n_control_messages++;
                for (auto& [cid, chunk] : local_chunks) {
                    for (size_t i = 0; i < chunk.n_messages(); ++i) {
                        outbound_chunks[chunk.part_id(i)]++;

                        ASSERT_FALSE(chunk.is_control_message(i));
                        n_data_messages++;
                        n_total_data_size += chunk.data_size(i);
                        n_total_metadata_size += chunk.metadata_size(i);
                    }
                }
            }
        }
        EXPECT_TRUE(shuffler.ready_postbox_.empty());

        EXPECT_EQ(outbound_chunks, shuffler.outbound_chunk_counter_);

        EXPECT_EQ(pids.size(), n_control_messages);
        EXPECT_EQ(pids.size(), n_data_messages);
        EXPECT_EQ(num_bytes * pids.size(), n_total_data_size);
        EXPECT_EQ(num_bytes * pids.size(), n_total_metadata_size);
    }

    std::vector<rapidsmpf::shuffler::PartID> pids;
    size_t num_bytes;
    std::shared_ptr<rapidsmpf::ProgressThread> progress_thread;
    std::unique_ptr<rapidsmpf::shuffler::Shuffler> shuffler;
    std::unique_ptr<rapidsmpf::BufferResource> br;
    rmm::cuda_stream_view stream;
};

TEST_P(ShuffleInsertGroupedTest, InsertPackedData) {
    // note: we disable periodic spill check to avoid the buffer resource from spilling
    // chunks in the ready postbox
    br = std::make_unique<rapidsmpf::BufferResource>(
        mr(),
        get_memory_available_map(rapidsmpf::MemoryType::DEVICE),
        std::nullopt  // disable periodic spill check
    );
    shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
        GlobalEnvironment->comm_, progress_thread, 0, pids.size(), stream, br.get()
    );

    // pause the progress thread to avoid extracting from outgoing_postbox_
    progress_thread->pause();

    auto chunks = generate_packed_data();
    shuffler->insert_grouped(std::move(chunks));
    shuffler->insert_finished(std::vector<rapidsmpf::shuffler::PartID>(pids));

    ASSERT_NO_FATAL_FAILURE(verify_shuffler_state(*shuffler));
}

TEST_P(ShuffleInsertGroupedTest, InsertPackedDataNoHeadroom) {
    // note: we disable periodic spill check to avoid the buffer resource from spilling
    // chunks in the ready postbox
    br = std::make_unique<rapidsmpf::BufferResource>(
        mr(),
        get_memory_available_map(rapidsmpf::MemoryType::HOST),
        std::nullopt  // disable periodic spill check
    );
    shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
        GlobalEnvironment->comm_, progress_thread, 0, pids.size(), stream, br.get()
    );

    // pause the progress thread to avoid extracting from outgoing_postbox_
    progress_thread->pause();

    auto chunks = generate_packed_data();
    shuffler->insert_grouped(std::move(chunks));
    shuffler->insert_finished(std::vector<rapidsmpf::shuffler::PartID>(pids));

    ASSERT_NO_FATAL_FAILURE(verify_shuffler_state(*shuffler));

    // resume progress thread - this will guarantee that shuffler progress function is
    // marked as done. This is important to ensure that the test does not hang.
    progress_thread->resume();
}

INSTANTIATE_TEST_SUITE_P(
    ShuffleTestP,
    ShuffleInsertGroupedTest,
    testing::Combine(
        testing::Values(1, 9, 100),  // total_num_partitions
        testing::Values(10, 1000)  // num_bytes
    ),
    [](const testing::TestParamInfo<ShuffleInsertGroupedTest::ParamType>& info) {
        return "total_nparts_" + std::to_string(std::get<0>(info.param)) + "__nbytes_"
               + std::to_string(std::get<1>(info.param));
    }
);

TEST(Shuffler, SpillOnInsertAndExtraction) {
    rapidsmpf::shuffler::PartID const total_num_partitions = 2;
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();

    // Use RapidsMPF's memory resource adaptor.
    rapidsmpf::RmmResourceAdaptor mr{cudf::get_current_device_resource_ref()};

    // Create a buffer resource with an availabe device memory we can control
    // through the variable `device_memory_available`.
    std::int64_t device_memory_available{0};
    rapidsmpf::BufferResource br{
        mr,
        {{rapidsmpf::MemoryType::DEVICE,
          [&device_memory_available]() -> std::int64_t { return device_memory_available; }
        }},
        std::nullopt  // disable periodic spill check
    };
    EXPECT_EQ(
        br.memory_available(rapidsmpf::MemoryType::DEVICE)(), device_memory_available
    );

    // Create a communicator of size 1, such that each shuffler will run locally.
    auto comm = GlobalEnvironment->split_comm();
    EXPECT_EQ(comm->nranks(), 1);

    std::shared_ptr<rapidsmpf::ProgressThread> progress_thread =
        std::make_shared<rapidsmpf::ProgressThread>(comm->logger());

    // Create a shuffler and input chunks.
    rapidsmpf::shuffler::Shuffler shuffler(
        comm,
        progress_thread,
        0,  // op_id
        total_num_partitions,
        stream,
        &br
    );
    cudf::table input_table = random_table_with_index(seed, 1000, 0, 10);
    auto input_chunks = rapidsmpf::partition_and_pack(
        input_table, {1}, total_num_partitions, hash_fn, seed, stream, mr
    );

    // Insert spills does nothing when device memory is available, we start
    // with 2 device allocations.
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);
    shuffler.insert(std::move(input_chunks));
    // And we end with two 2 device allocations.
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

    // Let's force spilling.
    device_memory_available = -1000;

    {
        // Now extract triggers spilling of the partition not being extracted.
        std::vector<rapidsmpf::PackedData> output_chunks = shuffler.extract(0);
        EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);

        // And insert also triggers spilling. We end up with zero device allocations.
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(0, std::move(output_chunks.at(0)));
        shuffler.insert(std::move(chunk));
        EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);
    }

    // Extract and unspill both partitions.
    std::vector<rapidsmpf::PackedData> out0 = shuffler.extract(0);
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 1);
    std::vector<rapidsmpf::PackedData> out1 = shuffler.extract(1);
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

    // Disable spilling and insert the first partition.
    device_memory_available = 1000;
    {
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(0, std::move(out0.at(0)));
        shuffler.insert(std::move(chunk));
    }
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 2);

    // Enable spilling and insert the second partition, which should trigger spilling
    // of both the first partition already in the shuffler and the second partition
    // that are being inserted.
    device_memory_available = -1000;
    {
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(1, std::move(out1.at(0)));
        shuffler.insert(std::move(chunk));
    }
    EXPECT_EQ(mr.get_main_record().num_current_allocs(), 0);

    shuffler.shutdown();
}

/**
 * @brief A test util that runs the wait test by first calling wait_fn lambda with no
 * partitions finished, and then with one partition finished. Former case, should timeout,
 * while the latter should pass. Since each wait function (wait, wait_on, wait_some) has
 * different output types, extract_pid_fn lambda is used to extract the pid from the
 * output.
 *
 * @tparam WaitFn a lambda that takes FinishCounter and PartID as arguments and returns
 * the result of the wait function.
 * @tparam ExctractPidFn a lambda that takes the result of WaitFn and returns the PID
 * extracted from the result.
 *
 * @param wait_fn wait lambda
 * @param extract_pid_fn extract partition ID lambda
 */
template <typename WaitFn, typename ExctractPidFn>
void run_wait_test(WaitFn&& wait_fn, ExctractPidFn&& extract_pid_fn) {
    rapidsmpf::shuffler::PartID out_nparts = 20;
    auto comm = GlobalEnvironment->comm_;

    if (comm->rank() != 0) {
        GTEST_SKIP() << "Test only runs on rank 0";
    }

    auto local_partitions = rapidsmpf::shuffler::Shuffler::local_partitions(
        comm, out_nparts, rapidsmpf::shuffler::Shuffler::round_robin
    );

    rapidsmpf::shuffler::detail::FinishCounter finish_counter(
        comm->nranks(), local_partitions
    );

    // pick some local partition to test
    auto p_id = local_partitions[0];

    // none of the partitions are finished now. So, wait_fn should timeout
    EXPECT_THROW(wait_fn(finish_counter, p_id), std::runtime_error);

    // move goalpost by 1 for the finished chunk msg
    finish_counter.move_goalpost(p_id, 1);
    for (auto i = 0; i < comm->nranks() - 1; i++) {
        // mark that no more chunks from other ranks by setting n_chunks=0
        finish_counter.move_goalpost(p_id, 0);
    }
    // add the finished chunk for partition p_id
    finish_counter.add_finished_chunk(p_id);

    // pass the wait_fn result to extract_pid_fn. It should return p_id
    EXPECT_EQ(p_id, extract_pid_fn(wait_fn(finish_counter, p_id)));
}

TEST(FinishCounterTests, wait_with_timeout) {
    ASSERT_NO_FATAL_FAILURE(run_wait_test(
        [](rapidsmpf::shuffler::detail::FinishCounter& finish_counter,
           rapidsmpf::shuffler::PartID const& /* exp_pid */) {
            return finish_counter.wait_any(std::chrono::milliseconds(10));
        },
        [](rapidsmpf::shuffler::PartID const p_id) { return p_id; }  // pass through
    ));
}

TEST(FinishCounterTests, wait_on_with_timeout) {
    ASSERT_NO_FATAL_FAILURE(run_wait_test(
        [&](rapidsmpf::shuffler::detail::FinishCounter& finish_counter,
            rapidsmpf::shuffler::PartID const& exp_pid) {
            finish_counter.wait_on(exp_pid, std::chrono::milliseconds(10));
            return exp_pid;  // return expected PID as wait_on return void
        },
        [&](rapidsmpf::shuffler::PartID const p_id) { return p_id; }  // pass through
    ));
}

TEST(FinishCounterTests, wait_some_with_timeout) {
    ASSERT_NO_FATAL_FAILURE(run_wait_test(
        [&](rapidsmpf::shuffler::detail::FinishCounter& finish_counter,
            rapidsmpf::shuffler::PartID const& /* exp_pid */) {
            return finish_counter.wait_some(std::chrono::milliseconds(10));
        },
        [&](std::vector<rapidsmpf::shuffler::PartID> const p_ids) {
            // extract the first element, as there will be only one finished partition
            return p_ids[0];
        }
    ));
}

namespace rapidsmpf::shuffler::detail {
Chunk make_dummy_chunk(ChunkID chunk_id, PartID part_id) {
    return Chunk(chunk_id, {part_id}, {0}, {0}, {0}, nullptr, nullptr);
}
}  // namespace rapidsmpf::shuffler::detail

class PostBoxTest : public cudf::test::BaseFixture {
  protected:
    using PostboxType = rapidsmpf::shuffler::detail::PostBox<rapidsmpf::Rank>;

    void SetUp() override {
        GlobalEnvironment->barrier();  // sync the env

        postbox = std::make_unique<PostboxType>(
            [this](rapidsmpf::shuffler::PartID part_id) {
                return partition_owner(part_id);
            },
            GlobalEnvironment->comm_->nranks()
        );
    }

    rapidsmpf::Rank partition_owner(rapidsmpf::shuffler::PartID part_id) {
        return rapidsmpf::shuffler::Shuffler::round_robin(
            GlobalEnvironment->comm_, part_id
        );
    }

    void TearDown() override {
        postbox.reset();
    }

    std::unique_ptr<PostboxType> postbox;
};

TEST_F(PostBoxTest, EmptyPostbox) {
    EXPECT_TRUE(postbox->empty());
    EXPECT_TRUE(postbox->extract_all_ready().empty());
}

TEST_F(PostBoxTest, InsertAndExtractMultipleChunks) {
    uint32_t const num_partitions =
        GlobalEnvironment->comm_->nranks() * 2;  // 2 paritions/ rank
    uint32_t const num_chunks = num_partitions * 4;  // 4 chunks/ partition

    // Insert chunks for rank 0
    for (uint32_t i = 0; i < num_chunks; ++i) {
        auto chunk = rapidsmpf::shuffler::detail::make_dummy_chunk(
            rapidsmpf::shuffler::detail::ChunkID{i},
            rapidsmpf::shuffler::PartID{i % num_partitions}
        );
        postbox->insert(std::move(chunk));
    }

    EXPECT_FALSE(postbox->empty());

    // extract chunks for each rank
    std::vector<rapidsmpf::shuffler::detail::Chunk> extracted_chunks;
    uint32_t extracted_nchunks = 0;
    for (rapidsmpf::Rank rank = 0; rank < GlobalEnvironment->comm_->nranks(); ++rank) {
        auto chunks = postbox->extract_by_key(rank);
        extracted_nchunks += chunks.size();

        for (auto& [_, chunk] : chunks) {
            extracted_chunks.emplace_back(std::move(chunk));
        }
    }
    EXPECT_EQ(extracted_nchunks, num_chunks);
    EXPECT_TRUE(postbox->empty());

    // reinsert the exctracted chunks
    for (auto& chunk : extracted_chunks) {
        postbox->insert(std::move(chunk));
    }

    // extract all chunks
    auto all_chunks = postbox->extract_all_ready();
    EXPECT_TRUE(postbox->empty());
    EXPECT_EQ(all_chunks.size(), num_chunks);
}

TEST_F(PostBoxTest, ThreadSafety) {
    constexpr uint32_t num_threads = 4;
    constexpr uint32_t chunks_per_thread = 100;
    constexpr uint32_t chunks_per_partition = 4;

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i] {
            for (uint32_t j = 0; j < chunks_per_thread; ++j) {
                auto chunk = rapidsmpf::shuffler::detail::make_dummy_chunk(
                    rapidsmpf::shuffler::detail::ChunkID{i * chunks_per_thread + j},
                    rapidsmpf::shuffler::PartID{j / chunks_per_partition}
                );
                postbox->insert(std::move(chunk));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all chunks were inserted correctly
    uint32_t extracted_nchunks = 0;
    for (rapidsmpf::Rank rank = 0; rank < GlobalEnvironment->comm_->nranks(); ++rank) {
        auto chunks = postbox->extract_by_key(rank);
        extracted_nchunks += chunks.size();
    }
    EXPECT_EQ(extracted_nchunks, num_threads * chunks_per_thread);

    EXPECT_TRUE(postbox->empty());
}

TEST(Shuffler, ShutdownWhilePaused) {
    auto stream = cudf::get_default_stream();
    auto progress_thread =
        std::make_shared<rapidsmpf::ProgressThread>(GlobalEnvironment->comm_->logger());
    auto mr = cudf::get_current_device_resource_ref();

    auto br = std::make_unique<rapidsmpf::BufferResource>(mr);

    auto shuffler = std::make_unique<rapidsmpf::shuffler::Shuffler>(
        GlobalEnvironment->comm_, progress_thread, 0, 1, stream, br.get()
    );

    // pause the progress thread to avoid extracting from outgoing_postbox_
    progress_thread->pause();

    // sleep this thread for 5ms, so that spill function is also run
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    EXPECT_FALSE(progress_thread->is_running());

    // shutdown shuffler while progress thread is paused
    shuffler->shutdown();
}
