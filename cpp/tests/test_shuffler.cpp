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
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
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

    auto chunks = rapidsmpf::shuffler::partition_and_pack(
        expect, {1}, num_partitions, hash_fn, seed, stream, mr
    );

    // Convert to a vector
    std::vector<rapidsmpf::PackedData> chunks_vector;
    for (auto& [_, chunk] : chunks) {
        chunks_vector.push_back(std::move(chunk));
    }
    EXPECT_EQ(chunks_vector.size(), num_partitions);

    auto result =
        rapidsmpf::shuffler::unpack_and_concat(std::move(chunks_vector), stream, mr);

    // Compare the input table with the result. We ignore the row order by
    // sorting by their index (first column).
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sort_table(expect), sort_table(result));
}

TEST(MetadataMessage, round_trip) {
    auto metadata = iota_vector<uint8_t>(100);

    rapidsmpf::shuffler::detail::Chunk expect(
        1, 2, true, 0, std::make_unique<std::vector<uint8_t>>(metadata), nullptr, nullptr
    );

    // Extract the metadata from then chunk.
    auto msg = expect.to_metadata_message();
    EXPECT_TRUE(expect.metadata->empty());

    // Create a new chunk from the message.
    auto result = rapidsmpf::shuffler::detail::Chunk::from_metadata_message(msg);

    // They should be identical.
    EXPECT_EQ(expect.pid, result.pid);
    EXPECT_EQ(expect.cid, result.cid);
    EXPECT_EQ(expect.expected_num_chunks, result.expected_num_chunks);
    EXPECT_EQ(expect.gpu_data, result.gpu_data);

    // The metadata should be identical to the original.
    EXPECT_EQ(metadata, *result.metadata);
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

void test_shuffler(
    std::shared_ptr<rapidsmpf::Communicator> const& comm,
    rapidsmpf::shuffler::Shuffler& shuffler,
    rapidsmpf::shuffler::PartID total_num_partitions,
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
    auto [expect_partitions, owner] = rapidsmpf::shuffler::partition_and_split(
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
            auto packed_chunks = rapidsmpf::shuffler::partition_and_pack(
                slice,
                {1},
                static_cast<std::int32_t>(total_num_partitions),
                hash_fn,
                seed,
                stream,
                mr
            );
            // Add the chunks to the shuffle
            shuffler.insert(std::move(packed_chunks));
        }
        row_offset += partiton_size;
    }
    // Tell the shuffler that we have no more input partitions.
    for (rapidsmpf::shuffler::PartID i = 0; i < total_num_partitions; ++i) {
        shuffler.insert_finished(i);
    }

    while (!shuffler.finished()) {
        auto finished_partition = shuffler.wait_any(wait_timeout);
        auto packed_chunks = shuffler.extract(finished_partition);
        auto result =
            rapidsmpf::shuffler::unpack_and_concat(std::move(packed_chunks), stream, mr);

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
    void SetUp() override {
        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }
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

TEST_P(MemoryAvailable_NumPartition, round_trip) {
    MemoryAvailableMap const memory_available = std::get<0>(GetParam());
    rapidsmpf::shuffler::PartID const total_num_partitions = std::get<1>(GetParam());
    std::size_t const total_num_rows = std::get<2>(GetParam());
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    rapidsmpf::BufferResource br{mr, memory_available};

    rapidsmpf::shuffler::Shuffler shuffler(
        GlobalEnvironment->comm_,
        GlobalEnvironment->progress_thread_,
        0,  // op_id
        total_num_partitions,
        stream,
        &br
    );

    EXPECT_NO_FATAL_FAILURE(test_shuffler(
        GlobalEnvironment->comm_,
        shuffler,
        total_num_partitions,
        total_num_rows,
        seed,
        hash_fn,
        stream,
        br.device_mr()
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
        br = std::make_shared<rapidsmpf::BufferResource>(mr());
        stream = cudf::get_default_stream();

        GlobalEnvironment->barrier();
    }

    void TearDown() override {
        GlobalEnvironment->barrier();
    }

    // test run for each thread. The test follows the same logic as
    // `MemoryAvailable_NumPartition` test, but without any memory limitations
    void RunTest(int t_id) {
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
            100,  // total_num_rows
            t_id,  // seed
            cudf::hash_id::HASH_MURMUR3,
            stream,
            br->device_mr()
        ));
    }

    int num_shufflers;
    rapidsmpf::shuffler::PartID total_num_partitions;

    std::shared_ptr<rapidsmpf::BufferResource> br;
    rmm::cuda_stream_view stream;
};

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

TEST_P(ConcurrentShuffleTest, round_trip) {
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(num_shufflers));

    for (int t_id = 0; t_id < num_shufflers; t_id++) {
        futures.push_back(std::async(std::launch::async, [this, t_id] {
            ASSERT_NO_FATAL_FAILURE(this->RunTest(t_id));
        }));
    }

    for (auto& f : futures) {
        ASSERT_NO_THROW(f.wait());
    }
}

TEST(Shuffler, SpillOnInsertAndExtraction) {
    rapidsmpf::shuffler::PartID const total_num_partitions = 2;
    std::int64_t const seed = 42;
    cudf::hash_id const hash_fn = cudf::hash_id::HASH_MURMUR3;
    auto stream = cudf::get_default_stream();

    // Use a statistics memory resource.
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> mr(
        cudf::get_current_device_resource_ref()
    );

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
    auto input_chunks = rapidsmpf::shuffler::partition_and_pack(
        input_table, {1}, total_num_partitions, hash_fn, seed, stream, mr
    );

    // Insert spills does nothing when device memory is available, we start
    // with 2 device allocations.
    EXPECT_EQ(mr.get_allocations_counter().value, 2);
    shuffler.insert(std::move(input_chunks));
    // And we end with two 2 device allocations.
    EXPECT_EQ(mr.get_allocations_counter().value, 2);

    // Let's force spilling.
    device_memory_available = -1000;

    {
        // Now extract triggers spilling of the partition not being extracted.
        std::vector<rapidsmpf::PackedData> output_chunks = shuffler.extract(0);
        EXPECT_EQ(mr.get_allocations_counter().value, 1);

        // And insert also triggers spilling. We end up with zero device allocations.
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(0, std::move(output_chunks.at(0)));
        shuffler.insert(std::move(chunk));
        EXPECT_EQ(mr.get_allocations_counter().value, 0);
    }

    // Extract and unspill both partitions.
    std::vector<rapidsmpf::PackedData> out0 = shuffler.extract(0);
    EXPECT_EQ(mr.get_allocations_counter().value, 1);
    std::vector<rapidsmpf::PackedData> out1 = shuffler.extract(1);
    EXPECT_EQ(mr.get_allocations_counter().value, 2);

    // Disable spilling and insert the first partition.
    device_memory_available = 1000;
    {
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(0, std::move(out0.at(0)));
        shuffler.insert(std::move(chunk));
    }
    EXPECT_EQ(mr.get_allocations_counter().value, 2);

    // Enable spilling and insert the second partition, which should trigger spilling
    // of both the first partition already in the shuffler and the second partition
    // that are being inserted.
    device_memory_available = -1000;
    {
        std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> chunk;
        chunk.emplace(1, std::move(out1.at(0)));
        shuffler.insert(std::move(chunk));
    }
    EXPECT_EQ(mr.get_allocations_counter().value, 0);

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
