#include <catch2/catch.hpp>
#include <thread>
#include <vector>

#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/buffer/buffer.hpp>

TEST_CASE("PostBoxByRank", "[shuffler]") {
    constexpr size_t NUM_RANKS = 4;
    PostBoxByRank postbox(NUM_RANKS);

    SECTION("Empty postbox") {
        REQUIRE(postbox.empty());
        REQUIRE(postbox.extract(0).empty());
        REQUIRE(postbox.extract_all().empty());
    }

    SECTION("Insert and extract single chunk") {
        Chunk chunk;
        chunk.pid = 1;
        chunk.cid = 42;
        chunk.gpu_data = std::make_unique<Buffer>(MemoryType::HOST, 100);

        postbox.insert(0, std::move(chunk));
        REQUIRE_FALSE(postbox.empty());

        auto chunks = postbox.extract(0);
        REQUIRE(chunks.size() == 1);
        REQUIRE(chunks[0].pid == 1);
        REQUIRE(chunks[0].cid == 42);
        REQUIRE(chunks[0].gpu_data->size == 100);
        REQUIRE(postbox.empty());
    }

    SECTION("Insert and extract multiple chunks") {
        // Insert chunks for rank 0
        for (int i = 0; i < 3; ++i) {
            Chunk chunk;
            chunk.pid = i;
            chunk.cid = i * 10;
            chunk.gpu_data = std::make_unique<Buffer>(MemoryType::HOST, 100);
            postbox.insert(0, std::move(chunk));
        }

        // Insert chunks for rank 1
        for (int i = 0; i < 2; ++i) {
            Chunk chunk;
            chunk.pid = i + 3;
            chunk.cid = (i + 3) * 10;
            chunk.gpu_data = std::make_unique<Buffer>(MemoryType::HOST, 100);
            postbox.insert(1, std::move(chunk));
        }

        REQUIRE_FALSE(postbox.empty());

        // Extract chunks for rank 0
        auto rank0_chunks = postbox.extract(0);
        REQUIRE(rank0_chunks.size() == 3);
        for (int i = 0; i < 3; ++i) {
            REQUIRE(rank0_chunks[i].pid == i);
            REQUIRE(rank0_chunks[i].cid == i * 10);
        }

        // Extract chunks for rank 1
        auto rank1_chunks = postbox.extract(1);
        REQUIRE(rank1_chunks.size() == 2);
        for (int i = 0; i < 2; ++i) {
            REQUIRE(rank1_chunks[i].pid == i + 3);
            REQUIRE(rank1_chunks[i].cid == (i + 3) * 10);
        }

        REQUIRE(postbox.empty());
    }

    SECTION("Extract all chunks") {
        // Insert chunks for multiple ranks
        for (int rank = 0; rank < 3; ++rank) {
            for (int i = 0; i < 2; ++i) {
                Chunk chunk;
                chunk.pid = rank * 10 + i;
                chunk.cid = rank * 100 + i;
                chunk.gpu_data = std::make_unique<Buffer>(MemoryType::HOST, 100);
                postbox.insert(rank, std::move(chunk));
            }
        }

        REQUIRE_FALSE(postbox.empty());

        auto all_chunks = postbox.extract_all();
        REQUIRE(all_chunks.size() == 3);  // One vector per rank

        // Verify chunks for each rank
        for (int rank = 0; rank < 3; ++rank) {
            REQUIRE(all_chunks[rank].size() == 2);
            for (int i = 0; i < 2; ++i) {
                REQUIRE(all_chunks[rank][i].pid == rank * 10 + i);
                REQUIRE(all_chunks[rank][i].cid == rank * 100 + i);
            }
        }

        REQUIRE(postbox.empty());
    }

    SECTION("Thread safety") {
        constexpr int NUM_THREADS = 4;
        constexpr int CHUNKS_PER_THREAD = 100;

        std::vector<std::thread> threads;
        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back([&postbox, i]() {
                for (int j = 0; j < CHUNKS_PER_THREAD; ++j) {
                    Chunk chunk;
                    chunk.pid = i;
                    chunk.cid = j;
                    chunk.gpu_data = std::make_unique<Buffer>(MemoryType::HOST, 100);
                    postbox.insert(i, std::move(chunk));
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Verify all chunks were inserted correctly
        for (int i = 0; i < NUM_THREADS; ++i) {
            auto chunks = postbox.extract(i);
            REQUIRE(chunks.size() == CHUNKS_PER_THREAD);
            for (int j = 0; j < CHUNKS_PER_THREAD; ++j) {
                REQUIRE(chunks[j].pid == i);
                REQUIRE(chunks[j].cid == j);
            }
        }

        REQUIRE(postbox.empty());
    }
} 