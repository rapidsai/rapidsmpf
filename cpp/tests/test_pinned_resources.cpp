/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/buffer/pinned_memory_resource.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/utils.hpp>

#include "utils.hpp"

class PinnedHostBufferTest : public ::testing::TestWithParam<size_t> {
  protected:
    void SetUp() override {
        if (rapidsmpf::is_pinned_memory_resources_supported()) {
            p_pool = std::make_unique<rapidsmpf::PinnedMemoryPool>(0);
            p_mr = std::make_shared<rapidsmpf::PinnedMemoryResource>(*p_pool);

            // check if the resource satisfies the resource concept
            [[maybe_unused]] rmm::host_async_resource_ref host_mr(*p_mr);
        } else {
            GTEST_SKIP() << "PinnedHostBuffer is not supported for CUDA versions "
                            "below " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR;
        }
    }

    void TearDown() override {
        p_mr.reset();
        p_pool.reset();
    }

    rmm::cuda_stream_view stream{};
    std::unique_ptr<rapidsmpf::PinnedMemoryPool> p_pool;
    std::shared_ptr<rapidsmpf::PinnedMemoryResource> p_mr;
    rmm::mr::cuda_async_memory_resource cuda_mr{};
};

// Test with various buffer sizes
INSTANTIATE_TEST_SUITE_P(
    VariableSizes,
    PinnedHostBufferTest,
    ::testing::Values(
        1,  // 1B
        1024,  // 1KB
        1048576  // 1MB
    ),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return std::to_string(info.param);
    }
);

TEST_P(PinnedHostBufferTest, synchronized_host_data) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto source_data = random_vector<uint8_t>(0, buffer_size);

    // Create pinned buffer using deep copy constructor
    rapidsmpf::PinnedHostBuffer buffer(source_data.data(), buffer_size, stream, p_mr);

    // Synchronize on stream to ensure copy is complete
    buffer.synchronize();

    // Check the contents using std::equal
    ASSERT_EQ(buffer.size(), buffer_size);
    ASSERT_NE(buffer.data(), nullptr);

    const auto* data = buffer.data();
    EXPECT_TRUE(
        std::equal(
            source_data.begin(), source_data.end(), reinterpret_cast<const uint8_t*>(data)
        )
    );

    // move constructor
    rapidsmpf::PinnedHostBuffer buffer2(std::move(buffer));
    // no need to synchronize because the stream is the same
    EXPECT_TRUE(
        std::equal(
            source_data.begin(),
            source_data.end(),
            reinterpret_cast<const uint8_t*>(buffer2.data())
        )
    );
    EXPECT_EQ(data, buffer2.data());

    // move assignment
    buffer = std::move(buffer2);
    // no need to synchronize because the stream is the same
    EXPECT_TRUE(
        std::equal(
            source_data.begin(),
            source_data.end(),
            reinterpret_cast<const uint8_t*>(buffer.data())
        )
    );
    EXPECT_EQ(data, buffer.data());

    // deep copy
    rapidsmpf::PinnedHostBuffer buffer3(buffer, p_mr);
    buffer3.synchronize();
    EXPECT_TRUE(
        std::equal(
            source_data.begin(),
            source_data.end(),
            reinterpret_cast<const uint8_t*>(buffer3.data())
        )
    );

    // Clean up
    buffer.deallocate_async();
    buffer2.deallocate_async();
    buffer3.deallocate_async();
}

TEST_P(PinnedHostBufferTest, device_data) {
    const size_t buffer_size = GetParam();

    // Create a vector with random data
    auto host_data = random_vector<uint8_t>(0, buffer_size);
    rmm::device_buffer dev_data(host_data.data(), buffer_size, stream, cuda_mr);

    // Create pinned buffer by copying device data on the same stream
    rapidsmpf::PinnedHostBuffer buffer(dev_data.data(), buffer_size, stream, p_mr);

    // Check the contents using std::equal
    ASSERT_EQ(buffer.size(), buffer_size);
    ASSERT_NE(buffer.data(), nullptr);

    buffer.synchronize();
    EXPECT_TRUE(
        std::equal(
            host_data.begin(),
            host_data.end(),
            reinterpret_cast<const uint8_t*>(buffer.data())
        )
    );
}

namespace {
rapidsmpf::PinnedHostBuffer stream_synchronized_copy(
    auto const& src,
    rmm::cuda_stream_view stream,
    std::shared_ptr<rapidsmpf::PinnedMemoryResource> mr
) {
    if (src.size() > 0) {
        // allocate a new buffer on the downstream stream
        rapidsmpf::PinnedHostBuffer ret(src.size(), stream, std::move(mr));
        // synchronize the downstream stream with upstream stream, so that src.data() is
        // ready to be copied.
        rapidsmpf::cuda_stream_join(stream, src.stream());
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpyAsync(ret.data(), src.data(), src.size(), cudaMemcpyDefault, stream)
        );
        // synchronize the upstream stream with the downstream stream, so that we complete
        // the async copy before src deallocation.
        rapidsmpf::cuda_stream_join(src.stream(), stream);
        return ret;
    }
    return rapidsmpf::PinnedHostBuffer(0, stream, std::move(mr));
}

template <typename SourceBufferT>
void stream_sync_copy_test(size_t buffer_size, auto& src_mr, auto& pinned_mr) {
    rmm::cuda_stream_pool stream_pool(2, rmm::cuda_stream::flags::non_blocking);
    auto stream1 = stream_pool.get_stream();
    auto stream2 = stream_pool.get_stream();

    auto host_data = random_vector<uint8_t>(0, buffer_size);

    // create a src buffer on stream1 with host data (blocking copy)
    SourceBufferT src_buf1(host_data.data(), buffer_size, stream1, src_mr);
    // create a src buffer on stream1 with the same data (non-blocking copy)
    SourceBufferT src_buf2;
    if constexpr (std::is_same_v<SourceBufferT, rmm::device_buffer>) {
        src_buf2 = SourceBufferT(src_buf1, stream1, src_mr);
    } else {
        src_buf2 = SourceBufferT(src_buf1, src_mr);
    }

    // create a pinned host buffer on stream2 with src_buf2 (non-blocking copy)
    auto pinned_buf1 = stream_synchronized_copy(src_buf2, stream2, pinned_mr);

    pinned_buf1.synchronize();
    EXPECT_TRUE(
        std::equal(
            host_data.begin(),
            host_data.end(),
            reinterpret_cast<const uint8_t*>(pinned_buf1.data())
        )
    );
}
}  // namespace

TEST_P(PinnedHostBufferTest, stream_synchronized_copy_rmm) {
    EXPECT_NO_FATAL_FAILURE(
        stream_sync_copy_test<rmm::device_buffer>(GetParam(), cuda_mr, p_mr)
    );
}

TEST_P(PinnedHostBufferTest, stream_synchronized_copy_pinned) {
    EXPECT_NO_FATAL_FAILURE(
        stream_sync_copy_test<rapidsmpf::PinnedHostBuffer>(GetParam(), p_mr, p_mr)
    );
}
