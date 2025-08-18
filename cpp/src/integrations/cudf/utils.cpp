/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>
#include <type_traits>

#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/dictionary.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

namespace {
struct str_cudf_column_scalar_fn {
    template <typename T>
        requires(cudf::is_numeric<T>())
    std::string operator()(
        cudf::column_view col,
        cudf::size_type index,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr
    ) {
        std::unique_ptr<cudf::scalar> scalar = cudf::get_element(col, index, stream, mr);
        auto typed_scalar = static_cast<cudf::numeric_scalar<T> const*>(scalar.get());
        T val = typed_scalar->value(stream);
        return std::to_string(val);
    }

    template <typename T>
        requires(!cudf::is_numeric<T>())
    std::string operator()(
        cudf::column_view /* col */,
        cudf::size_type /* index */,
        rmm::cuda_stream_view /* stream */,
        rmm::device_async_resource_ref /* mr */
    ) {
        RAPIDSMPF_FAIL("not implemented");
    }
};

struct cudf_column_data_size_fn {
    template <typename T>
        requires(cudf::is_fixed_width<T>())
    size_t operator()(cudf::column_view const& col, rmm::cuda_stream_view) {
        return static_cast<size_t>(col.size()) * cudf::size_of(col.type())
               + bitmask_size(col);
    }

    // string type specialization
    template <typename T>
        requires(std::is_same_v<T, cudf::string_view>)
    size_t operator()(cudf::column_view const& col, rmm::cuda_stream_view stream) {
        cudf::strings_column_view sv(col);
        return static_cast<size_t>(sv.chars_size(stream)) + bitmask_size(col);
    }

    // compound type specialization except string
    template <typename T>
        requires(!std::is_same_v<T, cudf::string_view> && cudf::is_compound<T>())
    size_t operator()(cudf::column_view const& col, rmm::cuda_stream_view) {
        // compound types (except string) ie. list, dict, structs dont have a
        // content::data buffer. Data is stored in children columns. So, just return the
        // bitmask size.
        return bitmask_size(col);
    }

    template <typename T>
    size_t operator()(cudf::column_view const& col, rmm::cuda_stream_view) {
        RAPIDSMPF_FAIL("not implemented for type: " + cudf::type_to_name(col.type()));
    }

    static size_t bitmask_size(cudf::column_view const& col) {
        return col.has_nulls() ? cudf::bitmask_allocation_size_bytes(col.size()) : 0;
    }
};

}  // namespace

std::string str(
    cudf::column_view col,
    cudf::size_type index,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    return cudf::type_dispatcher(
        col.type(), str_cudf_column_scalar_fn{}, col, index, stream, mr
    );
}

std::string str(
    cudf::column_view col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr
) {
    std::stringstream ss;
    ss << "Column([";
    for (cudf::size_type i = 0; i < col.size(); ++i) {
        ss << str(col, i, stream, mr) << ", ";
    }
    ss << (col.size() == 0 ? "])" : "\b\b])");
    return ss.str();
}

std::string str(
    cudf::table_view tbl, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr
) {
    std::stringstream ss;
    ss << "Table([";
    for (auto col : tbl) {
        ss << str(col, stream, mr) << ", ";
    }
    ss << (tbl.num_columns() == 0 ? "])" : "\b\b])");
    return ss.str();
}

/**
 * @brief Calculate the memory usage of a column.
 *
 * @param col The column to calculate the memory usage of.
 * @return The memory usage of the column.
 */
size_t estimated_memory_usage(
    cudf::column_view const& col, rmm::cuda_stream_view stream
) {
    return std::transform_reduce(
        col.child_begin(),
        col.child_end(),
        cudf::type_dispatcher(col.type(), cudf_column_data_size_fn{}, col, stream),
        std::plus{},
        [&stream](cudf::column_view const& child) {
            return estimated_memory_usage(child, stream);
        }
    );
}

/**
 * @brief Calculate the memory usage of a table.
 *
 * @param tbl The table to calculate the memory usage of.
 * @return The memory usage of the table.
 */
size_t estimated_memory_usage(cudf::table_view const& tbl, rmm::cuda_stream_view stream) {
    return std::transform_reduce(
        tbl.begin(),
        tbl.end(),
        size_t{0},
        std::plus{},
        [&stream](cudf::column_view const& col) {
            return estimated_memory_usage(col, stream);
        }
    );
}

}  // namespace rapidsmpf
