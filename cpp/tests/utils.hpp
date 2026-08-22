/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include <cuda/memory_resource>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/packed_data.hpp>

/**
 * @brief RAII temporary directory created under GTest's temp directory.
 *
 * The directory is created on construction and recursively removed on
 * destruction. Removal errors are ignored.
 */
class TempDir {
  public:
    TempDir() : path_(unique_path()) {
        std::error_code ec;
        if (!std::filesystem::create_directories(path_, ec) || ec) {
            throw std::runtime_error(
                "Failed to create temp directory: " + path_.string()
            );
        }
    }

    ~TempDir() noexcept {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        // Intentionally ignore errors in destructor.
    }

    TempDir(TempDir const&) = delete;
    TempDir& operator=(TempDir const&) = delete;
    TempDir(TempDir&&) = delete;
    TempDir& operator=(TempDir&&) = delete;

    /// @brief Returns the path to the temporary directory.
    [[nodiscard]] std::filesystem::path const& path() const noexcept {
        return path_;
    }

  private:
    static std::filesystem::path unique_path() {
        static std::atomic<std::uint64_t> counter{0};
        return std::filesystem::path(testing::TempDir())
               / ("tmp_" + std::to_string(::getpid()) + "_"
                  + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
    }

    std::filesystem::path path_;
};

/// @brief User-defined literal for specifying memory sizes in KiB.
constexpr std::size_t operator"" _KiB(unsigned long long val) {
    return val * (1 << 10);
}

/// @brief User-defined literal for specifying memory sizes in MiB.
constexpr std::size_t operator"" _MiB(unsigned long long val) {
    return val * (1ull << 20);
}

/// @brief User-defined literal for specifying memory sizes in GiB.
constexpr std::size_t operator"" _GiB(unsigned long long val) {
    return val * (1 << 30);
}

template <typename T>
[[nodiscard]] std::vector<T> iota_vector(std::size_t nelem, T start = 0) {
    std::vector<T> ret(nelem);
    std::iota(ret.begin(), ret.end(), start);
    return ret;
}

template <typename T>
[[nodiscard]] inline std::vector<T> random_vector(
    std::int64_t seed,
    std::size_t nelem,
    T min = std::numeric_limits<T>::min(),
    T max = std::numeric_limits<T>::max()
) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<T> dist(min, max);
    std::vector<T> ret(nelem);
    std::generate(ret.begin(), ret.end(), [&]() { return dist(rng); });
    return ret;
}

/// @brief Create a PackedData object from a host buffer
[[nodiscard]] inline rapidsmpf::PackedData create_packed_data(
    std::span<std::uint8_t const> metadata,
    std::span<std::uint8_t const> data,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource* br
) {
    auto metadata_ptr =
        std::make_unique<std::vector<std::uint8_t>>(metadata.begin(), metadata.end());

    auto reservation = br->reserve(
        rapidsmpf::MemoryType::DEVICE, data.size(), rapidsmpf::AllowOverbooking::YES
    );
    auto data_ptr =
        std::make_unique<rmm::device_buffer>(data.data(), data.size(), stream);
    return rapidsmpf::PackedData{
        std::move(metadata_ptr), br->move(std::move(data_ptr), stream)
    };
}

/**
 * @brief Generate a packed data object with the given number of elements and offset.
 *
 * Both metadata and GPU data contain the same integer sequence of type T.
 *
 * @tparam T Element type stored in the buffer (default: int).
 * @param n_elements Number of elements in the sequence.
 * @param offset Starting value of the sequence.
 * @param stream CUDA stream for device allocation.
 * @param br Buffer resource used for allocations.
 * @return A packed data object containing metadata and GPU data.
 */
template <typename T = int>
[[nodiscard]] inline rapidsmpf::PackedData generate_packed_data(
    std::size_t n_elements,
    T offset,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br,
    rapidsmpf::AllowOverbooking allow_overbooking = rapidsmpf::AllowOverbooking::YES
) {
    auto const values = iota_vector<T>(n_elements, offset);
    auto const* bytes = reinterpret_cast<std::uint8_t const*>(values.data());

    auto metadata = std::make_unique<std::vector<std::uint8_t>>(
        bytes, bytes + values.size() * sizeof(T)
    );
    auto [reservation, _] =
        br.reserve(rapidsmpf::MemoryType::DEVICE, metadata->size(), allow_overbooking);
    auto data = br.make_buffer(stream, std::move(reservation));

    data->write_access([d_ptr = metadata->data(), m_size = metadata->size()](
                           std::byte* ptr, rmm::cuda_stream_view op_stream
                       ) {
        RAPIDSMPF_CUDA_TRY(rapidsmpf::cuda_memcpy_async(ptr, d_ptr, m_size, op_stream));
    });

    return {std::move(metadata), std::move(data)};
}

/**
 * @brief Validate a packed data object by checking metadata and GPU data contents.
 *
 * @tparam T Element type stored in the buffer (default: int).
 * @param packed_data Packed data object to validate.
 * @param n_elements Expected number of elements.
 * @param offset Expected starting value of the sequence.
 * @param stream CUDA stream used for device-host transfers.
 * @param br Buffer resource used for host allocation.
 */
template <typename T = int>
inline void validate_packed_data(
    rapidsmpf::PackedData&& packed_data,
    std::size_t n_elements,
    T offset,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br
) {
    auto const& metadata = *packed_data.metadata;
    EXPECT_EQ(n_elements * sizeof(T), metadata.size());

    for (std::size_t i = 0; i < n_elements; i++) {
        T val;
        std::memcpy(&val, metadata.data() + i * sizeof(T), sizeof(T));
        EXPECT_EQ(offset + static_cast<T>(i), val);
    }

    EXPECT_EQ(n_elements * sizeof(T), packed_data.data->size);

    auto res = br.reserve_or_fail(packed_data.data->size, rapidsmpf::MemoryType::HOST);
    auto data_on_host = br.move_to_host_buffer(std::move(packed_data.data), res);
    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    EXPECT_EQ(metadata, data_on_host->copy_to_uint8_vector());
}

/**
 * @brief Device memory resource that can inject stream-ordered delays.
 *
 * When enabled, each allocation enqueues a host callback on the allocation
 * stream that sleeps for a configurable duration. This blocks the CUDA stream
 * (making `cudaEventQuery` return not-ready) without blocking the host thread,
 * so the progress thread's event loop continues to run while data buffers
 * appear unready.
 */
class DelayedMemoryResource {
  public:
    DelayedMemoryResource(
        rmm::device_async_resource_ref upstream, std::chrono::milliseconds delay
    )
        : upstream_{upstream}, delay_{delay} {}

    void* allocate_sync(std::size_t, std::size_t) {
        RAPIDSMPF_FAIL("synchronous allocation not supported", std::invalid_argument);
    }

    void deallocate_sync(void*, std::size_t, std::size_t) noexcept {
        RAPIDSMPF_FATAL("synchronous deallocation not supported");
    }

    void* allocate(
        rmm::cuda_stream_view stream,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) {
        void* ptr = upstream_.allocate(stream, size, alignment);
        if (size > 0) {
            RAPIDSMPF_CUDA_TRY(cudaLaunchHostFunc(
                stream.value(), sleep_on_stream, new std::chrono::milliseconds(delay_)
            ));
        }
        return ptr;
    }

    void deallocate(
        rmm::cuda_stream_view stream,
        void* ptr,
        std::size_t size,
        std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT
    ) noexcept {
        upstream_.deallocate(stream, ptr, size, alignment);
    }

    bool operator==(DelayedMemoryResource const& other) const noexcept {
        return this == &other;
    }

    bool operator!=(DelayedMemoryResource const& other) const noexcept {
        return !(this == &other);
    }

    friend void get_property(
        DelayedMemoryResource const&, cuda::mr::device_accessible
    ) noexcept {}

  private:
    static void CUDART_CB sleep_on_stream(void* user_data) {
        auto* delay = static_cast<std::chrono::milliseconds*>(user_data);
        std::this_thread::sleep_for(*delay);
        delete delay;
    }

    cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
    std::chrono::milliseconds delay_;
};

static_assert(cuda::mr::resource<DelayedMemoryResource>);
static_assert(
    cuda::mr::resource_with<DelayedMemoryResource, cuda::mr::device_accessible>
);
