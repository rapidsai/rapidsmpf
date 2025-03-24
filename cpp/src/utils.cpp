/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>

#include <cudf/copying.hpp>

#include <rapidsmp/error.hpp>
#include <rapidsmp/utils.hpp>

namespace rapidsmp {

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
        RAPIDSMP_FAIL("not implemented");
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

#if __has_include(<valgrind/valgrind.h>)
#include <valgrind/valgrind.h>

bool is_running_under_valgrind() {
    static bool ret = RUNNING_ON_VALGRIND;
    return ret;
}
#else
bool is_running_under_valgrind() {
    return false;
}
#endif

std::string trim(std::string const& str) {
    std::stringstream trimmer;
    trimmer << str;
    std::string ret;
    trimmer >> ret;
    return ret;
}

std::string to_lower(std::string str) {
    // Special considerations regarding the case conversion:
    // - std::tolower() is not an addressable function. Passing it to std::transform()
    //   as a function pointer, if the compile turns out successful, causes the program
    //   behavior "unspecified (possibly ill-formed)", hence the lambda. ::tolower() is
    //   addressable and does not have this problem, but the following item still applies.
    // - To avoid UB in std::tolower() or ::tolower(), the character must be cast to
    // unsigned char.
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return str;
}

std::string to_upper(std::string str) {
    // Special considerations regarding the case conversion, see to_lower().
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    return str;
}

}  // namespace rapidsmp
