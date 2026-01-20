/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>


using namespace rapidsmpf;

TEST(CudaStreamJoinCppOnly, MultiUpstreamsMultiDownstreams) {
    // 3 upstreams write disjoint slices repeatedly (slow), then 3 downstreams
    // overwrite those slices once (fast). With priority streams, missing the join
    // is even more likely to produce the wrong final contents.
    constexpr int num_slices = 3;
    constexpr int upstream_repeats = 1000;

    const std::array<unsigned char, num_slices> upstream_values{0x11, 0x22, 0x33};
    const auto downstream_value = [](int i) {
        return static_cast<unsigned char>(0xE0 + i);
    };

    // Streams and their views (created with explicit priorities).
    std::array<cudaStream_t, num_slices> upstream_raw{};
    std::array<cudaStream_t, num_slices> downstream_raw{};
    std::array<rmm::cuda_stream_view, num_slices> upstreams{};
    std::array<rmm::cuda_stream_view, num_slices> downstreams{};

    int least_priority = 0;  // numerically larger (often 0) => lower priority
    int greatest_priority = 0;  // numerically smaller (often negative) => higher priority
    RAPIDSMPF_CUDA_TRY(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority)
    );

    for (int i = 0; i < num_slices; ++i) {
        RAPIDSMPF_CUDA_TRY(cudaStreamCreateWithPriority(
            &upstream_raw[i], cudaStreamNonBlocking, least_priority
        ));  // low priority
        RAPIDSMPF_CUDA_TRY(cudaStreamCreateWithPriority(
            &downstream_raw[i], cudaStreamNonBlocking, greatest_priority
        ));  // high priority
        upstreams[i] = rmm::cuda_stream_view{upstream_raw[i]};
        downstreams[i] = rmm::cuda_stream_view{downstream_raw[i]};
    }

    // One large device buffer, initialize to 0x00 and sync once for known base state.
    constexpr size_t total_bytes = 1 << 25;
    rmm::device_buffer buf(total_bytes, upstreams[0]);
    RAPIDSMPF_CUDA_TRY(cudaMemset(buf.data(), 0x00, buf.size()));
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());

    constexpr size_t slice_bytes = total_bytes / num_slices;
    ASSERT_GT(slice_bytes, 0u);
    auto* dptr = static_cast<unsigned char*>(buf.data());

    // Upstreams: each writes its slice repeatedly to stretch execution time.
    for (int i = 0; i < num_slices; ++i) {
        unsigned char* slice_dev_ptr = dptr + size_t(i) * slice_bytes;
        for (int r = 0; r < upstream_repeats; ++r) {
            RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(
                slice_dev_ptr, upstream_values[i], slice_bytes, upstreams[i]
            ));
        }
    }

    // Join: all downstreams wait on all upstreams.
    cuda_stream_join(downstreams, upstreams);

    // Downstreams: single, quick overwrite per slice (should win if join is correct).
    for (int i = 0; i < num_slices; ++i) {
        unsigned char* slice_dev_ptr = dptr + size_t(i) * slice_bytes;
        RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(
            slice_dev_ptr, downstream_value(i), slice_bytes, downstreams[i]
        ));
    }

    // Complete all work.
    RAPIDSMPF_CUDA_TRY(cudaDeviceSynchronize());

    // Verify: each slice must equal its downstream value (0xE0 + i).
    for (int i = 0; i < num_slices; ++i) {
        std::vector<unsigned char> host(slice_bytes);
        unsigned char const* slice_dev_ptr = dptr + size_t(i) * slice_bytes;

        RAPIDSMPF_CUDA_TRY(
            cudaMemcpy(host.data(), slice_dev_ptr, slice_bytes, cudaMemcpyDeviceToHost)
        );

        unsigned char expect = downstream_value(i);
        for (size_t j = 0; j < slice_bytes; ++j) {
            ASSERT_EQ(host[j], expect) << "slice " << i << " byte " << j << " mismatch";
        }
    }

    // Cleanup streams
    for (int i = 0; i < num_slices; ++i) {
        if (upstream_raw[i]) {
            RAPIDSMPF_CUDA_TRY(cudaStreamDestroy(upstream_raw[i]));
        }
        if (downstream_raw[i]) {
            RAPIDSMPF_CUDA_TRY(cudaStreamDestroy(downstream_raw[i]));
        }
    }
}
