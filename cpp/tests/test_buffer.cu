/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <thrust/equal.h>

#include <rmm/exec_policy.hpp>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/buffer/resource.hpp>

namespace rapidsmp {


TEST(AFoo, use_rmm_device_only) {
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    static constexpr std::initializer_list<uint8_t> dummy_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    static constexpr size_t len = dummy_data.size();
    auto dev_buf =
        rmm::device_buffer(len, stream, cudf::get_current_device_resource_ref());

    RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
        dev_buf.data(), std::data(dummy_data), len, cudaMemcpyHostToDevice, stream
    ));

    EXPECT_TRUE(
        thrust::equal(
            rmm::exec_policy(rmm::cuda_stream_default),
            static_cast<cuda::std::byte const*>(dev_buf.data()),
            static_cast<cuda::std::byte const*>(dev_buf.data()) + 8,
            static_cast<cuda::std::byte const*>(dev_buf.data())
        )
    );
}

constexpr std::size_t operator""_KiB(unsigned long long n) {
    return n * (1 << 10);
}

// Allocate a buffer of the given size from the given resource.
std::unique_ptr<Buffer> allocate_buffer(
    MemoryType mem_type,
    std::size_t size,
    BufferResource& br,
    rmm::cuda_stream_view stream
) {
    auto [res, _] = br.reserve(mem_type, size, false);
    return br.allocate(mem_type, size, stream, res);
}

class Foo : public testing::Test {
  public:
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;
    static constexpr std::initializer_list<uint8_t> dummy_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    static constexpr size_t len = dummy_data.size();

    std::unique_ptr<Buffer> data_buf;
    BufferResource br{cudf::get_current_device_resource_ref(), {{MemoryType::DEVICE, [] {
                                                                     return 1000_KiB;
                                                                 }}}};

    Foo() {
        data_buf = allocate_buffer(MemoryType::DEVICE, len, br, stream);
        RAPIDSMP_CUDA_TRY_ALLOC(cudaMemcpyAsync(
            data_buf->data(), std::data(dummy_data), len, cudaMemcpyHostToDevice, stream
        ));
    }
};

TEST_F(Foo, temp1) {
    ASSERT_EQ(data_buf->mem_type(), MemoryType::DEVICE);
    ASSERT_EQ(const_cast<Buffer const&>(*data_buf).device()->ssize(), len);

    Buffer const& data_buf1 = *data_buf;

    EXPECT_TRUE(
        thrust::equal(
            rmm::exec_policy(rmm::cuda_stream_default),
            static_cast<cuda::std::byte const*>(data_buf1.device()->data()),
            static_cast<cuda::std::byte const*>(data_buf1.device()->data()) + len,
            static_cast<cuda::std::byte const*>(data_buf1.device()->data())
        )
    );
}

TEST_F(Foo, temp2) {
    ASSERT_EQ(data_buf->mem_type(), MemoryType::DEVICE);
    ASSERT_EQ(const_cast<Buffer const&>(*data_buf).device()->size(), len);

    Buffer const& data_buf1 = *data_buf;

    EXPECT_TRUE(
        thrust::equal(
            rmm::exec_policy(rmm::cuda_stream_default),
            const_cast<char*>(static_cast<const char*>(data_buf1.data())),
            const_cast<char*>(static_cast<const char*>(data_buf1.data())) + len,
            const_cast<char*>(static_cast<const char*>(data_buf1.data()))
        )
    );
}

}  // namespace rapidsmp
