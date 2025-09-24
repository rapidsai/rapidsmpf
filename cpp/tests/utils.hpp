/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rapidsmpf/buffer/packed_data.hpp>
#include <rapidsmpf/error.hpp>

/**
 * @brief User-defined literal for specifying memory sizes in MiB.
 */
constexpr std::size_t operator"" _MiB(unsigned long long val) {
    return val * (1ull << 20);
}

template <typename T>
[[nodiscard]] std::vector<T> iota_vector(std::size_t nelem, T start = 0) {
    std::vector<T> ret(nelem);
    std::iota(ret.begin(), ret.end(), start);
    return ret;
}

template <typename T>
[[nodiscard]] inline std::unique_ptr<cudf::column> iota_column(
    std::size_t nrows, T start = 0
) {
    std::vector<T> vec = iota_vector(nrows, start);
    cudf::test::fixed_width_column_wrapper<T> ret(vec.begin(), vec.end());
    return ret.release();
}

[[nodiscard]] inline std::vector<std::int64_t> random_vector(
    std::int64_t seed,
    std::size_t nelem,
    std::int64_t min = std::numeric_limits<std::int64_t>::min(),
    std::int64_t max = std::numeric_limits<std::int64_t>::max()
) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<std::int64_t> dist(min, max);
    std::vector<std::int64_t> ret(nelem);
    std::generate(ret.begin(), ret.end(), [&]() { return dist(rng); });
    return ret;
}

[[nodiscard]] inline std::unique_ptr<cudf::column> random_column(
    std::int64_t seed,
    std::size_t nrows,
    std::int64_t min = std::numeric_limits<std::int64_t>::min(),
    std::int64_t max = std::numeric_limits<std::int64_t>::max()
) {
    std::vector<std::int64_t> vec = random_vector(seed, nrows, min, max);
    cudf::test::fixed_width_column_wrapper<std::int64_t> ret(vec.begin(), vec.end());
    return ret.release();
}

[[nodiscard]] inline cudf::table random_table_with_index(
    std::int64_t seed,
    std::size_t nrows,
    std::int64_t min = std::numeric_limits<std::int64_t>::min(),
    std::int64_t max = std::numeric_limits<std::int64_t>::max()
) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(iota_column<std::int64_t>(nrows));
    cols.push_back(random_column(seed, nrows, min, max));
    return cudf::table(std::move(cols));
}

[[nodiscard]] inline cudf::table sort_table(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& /* column_indices */ = {0}
) {
    return cudf::gather(table, cudf::sorted_order(table.select({0}))->view())->release();
}

[[nodiscard]] inline cudf::table sort_table(
    std::unique_ptr<cudf::table> const& table,
    std::vector<cudf::size_type> const& column_indices = {0}
) {
    return sort_table(table->view(), column_indices);
}

/// @brief Create a PackedData object from a host buffer
[[nodiscard]] inline rapidsmpf::PackedData create_packed_data(
    std::span<uint8_t const> metadata,
    std::span<uint8_t const> data,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource* br
) {
    auto metadata_ptr =
        std::make_unique<std::vector<uint8_t>>(metadata.begin(), metadata.end());

    auto reservation = br->reserve(rapidsmpf::MemoryType::DEVICE, data.size(), true);
    auto data_ptr =
        std::make_unique<rmm::device_buffer>(data.data(), data.size(), stream);
    return rapidsmpf::PackedData{
        std::move(metadata_ptr), br->move(std::move(data_ptr), stream)
    };
}

/**
 * @brief Generate a packed data object with the given number of elements and offset.
 *
 * Both metadata and GPU data contain the same integer sequence.
 *
 * @param n_elements Number of elements in the sequence.
 * @param offset Starting value of the sequence.
 * @param stream CUDA stream for device allocation.
 * @param br Buffer resource used for allocations.
 * @return A packed data object containing metadata and GPU data.
 */
[[nodiscard]] inline rapidsmpf::PackedData generate_packed_data(
    int n_elements,
    int offset,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br
) {
    auto values = iota_vector<int>(n_elements, offset);

    auto metadata = std::make_unique<std::vector<uint8_t>>(n_elements * sizeof(int));
    std::memcpy(metadata->data(), values.data(), n_elements * sizeof(int));

    auto data = std::make_unique<rmm::device_buffer>(
        values.data(), n_elements * sizeof(int), stream, br.device_mr()
    );

    return {std::move(metadata), br.move(std::move(data), stream)};
}

/**
 * @brief Validate a packed data object by checking metadata and GPU data contents.
 *
 * @param packed_data Packed data object to validate.
 * @param n_elements Expected number of elements.
 * @param offset Expected starting value of the sequence.
 * @param stream CUDA stream used for device-host transfers.
 * @param br Buffer resource used for host allocation.
 */
inline void validate_packed_data(
    rapidsmpf::PackedData&& packed_data,
    int n_elements,
    int offset,
    rmm::cuda_stream_view stream,
    rapidsmpf::BufferResource& br
) {
    auto const& metadata = *packed_data.metadata;
    EXPECT_EQ(n_elements * sizeof(int), metadata.size());

    for (int i = 0; i < n_elements; i++) {
        int val;
        std::memcpy(&val, metadata.data() + i * sizeof(int), sizeof(int));
        EXPECT_EQ(offset + i, val);
    }

    EXPECT_EQ(n_elements * sizeof(int), packed_data.data->size);
    auto copied_vec = br.allocate(
        stream, br.reserve_or_fail(n_elements * sizeof(int), rapidsmpf::MemoryType::HOST)
    );
    rapidsmpf::buffer_copy(*copied_vec, *packed_data.data, n_elements * sizeof(int));
    RAPIDSMPF_CUDA_TRY(cudaStreamSynchronize(stream));
    EXPECT_EQ(metadata, *const_cast<rapidsmpf::Buffer const&>(*copied_vec).host());
}
