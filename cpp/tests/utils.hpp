/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/column_wrapper.hpp>

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
