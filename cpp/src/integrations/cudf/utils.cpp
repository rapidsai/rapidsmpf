/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/copying.hpp>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/utils.hpp>

namespace rapidsmpf {

namespace {
struct str_cudf_column_scalar_fn {
    template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
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

    template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
    std::string operator()(
        cudf::column_view /* col */,
        cudf::size_type /* index */,
        rmm::cuda_stream_view /* stream */,
        rmm::device_async_resource_ref /* mr */
    ) {
        RAPIDSMPF_FAIL("not implemented");
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

}  // namespace rapidsmpf
