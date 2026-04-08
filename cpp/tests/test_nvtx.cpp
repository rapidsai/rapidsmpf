/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <string_view>
#include <thread>

#include <gtest/gtest.h>

#include <rapidsmpf/nvtx.hpp>

// ── extract_func_name ────────────────────────────────────────────────────────

using rapidsmpf::detail::extract_func_name;

TEST(ExtractFuncNameTest, VariousCases) {
    // return type + namespace + function name + params
    EXPECT_EQ(extract_func_name("void rapidsmpf::baz(int)"), "rapidsmpf::baz");

    // typical GCC/Clang source_location::function_name() for a class method
    EXPECT_EQ(extract_func_name("void rapidsmpf::Foo::bar(int)"), "rapidsmpf::Foo::bar");

    // constructors have no return type so there is no leading space
    EXPECT_EQ(extract_func_name("rapidsmpf::Foo::Foo(int)"), "rapidsmpf::Foo::Foo");

    // const qualifier appears after ')' and must not affect the extracted name
    EXPECT_EQ(
        extract_func_name("int rapidsmpf::Foo::get() const"), "rapidsmpf::Foo::get"
    );

    // plain free function without any namespace prefix
    EXPECT_EQ(extract_func_name("void bar(float, double)"), "bar");

    // empty parameter list
    EXPECT_EQ(extract_func_name("void Foo::bar()"), "Foo::bar");

    // GCC appends "[with T = int]" after the closing ')'; the first '(' is still
    // the one opening the parameter list, so the result must stay clean
    EXPECT_EQ(extract_func_name("void Foo::bar(T) [with T = int]"), "Foo::bar");

    // source_location::function_name() for a lambda shows the outer function's
    // name followed by "::<lambda(...)>".  extract_func_name should return the
    // outer function portion (up to the first '('), which is more useful than
    // reporting "operator()".
    EXPECT_EQ(
        extract_func_name("auto rapidsmpf::Foo::method()::<lambda()>"),
        "rapidsmpf::Foo::method"
    );

    // function name without paranthesis
    EXPECT_EQ(extract_func_name("void Foo::bar"), "Foo::bar");
}

// ── RAPIDSMPF_NVTX_FUNC_RANGE smoke tests ───────────────────────────────────
//
// Profile with:
// nsys profile -o out -t nvtx ./cpp/build/gtests/single_tests --gtest_filter="*Nvtx*"

// Helper class used to exercise the macro in a member-function context.
struct NvtxSmokeHelper {
    void method() {
        RAPIDSMPF_NVTX_FUNC_RANGE();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    static void static_method() {
        RAPIDSMPF_NVTX_FUNC_RANGE();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void method_with_payload(int n) {
        RAPIDSMPF_NVTX_FUNC_RANGE(n);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

TEST(NvtxFuncRangeTest, SmokeFreeFunction) {
    // Verifies the macro compiles and does not crash for a free function.
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    SUCCEED();
}

TEST(NvtxFuncRangeTest, SmokeFreeFunctionWithPayload) {
    RAPIDSMPF_NVTX_FUNC_RANGE(42);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    SUCCEED();
}

TEST(NvtxFuncRangeTest, SmokeMemberFunction) {
    NvtxSmokeHelper::static_method();
    NvtxSmokeHelper h;
    h.method();
    h.method_with_payload(7);
    SUCCEED();
}
