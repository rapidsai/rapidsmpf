/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <rapidsmpf/error.hpp>

/*
 * This file contains tests for the RAPIDSMPF error macros.
 *
 * RAPIDSMPF macros are not public API and should not be used externally, but we
 * test them to avoid regressions anyway.
 *
 * The macros are tested for:
 * - Successful operations (should not throw)
 * - Failed operations (should throw the appropriate exception)
 * - Error message formatting
 * - Actual CUDA operations
 */

// Test RAPIDSMPF_EXPECTS macro with condition that evaluates to true (should not throw)
TEST(ErrorMacrosTest, ExpectsNoThrow) {
    EXPECT_NO_THROW(RAPIDSMPF_EXPECTS(true, "This should not throw"));
    EXPECT_NO_THROW(RAPIDSMPF_EXPECTS(true, "This should not throw", std::runtime_error));
}

// Test RAPIDSMPF_EXPECTS macro with condition that evaluates to false (should throw)
TEST(ErrorMacrosTest, ExpectsThrow) {
    EXPECT_THROW(RAPIDSMPF_EXPECTS(false, "Expected exception"), std::logic_error);
    EXPECT_THROW(
        RAPIDSMPF_EXPECTS(false, "Expected runtime error", std::runtime_error),
        std::runtime_error
    );
}

// Test RAPIDSMPF_FAIL macro (should always throw)
TEST(ErrorMacrosTest, FailThrow) {
    EXPECT_THROW(RAPIDSMPF_FAIL("This should throw logic_error"), std::logic_error);
    EXPECT_THROW(
        RAPIDSMPF_FAIL("This should throw runtime_error", std::runtime_error),
        std::runtime_error
    );
}

// Test RAPIDSMPF_CUDA_TRY macro with successful CUDA call (should not throw)
TEST(ErrorMacrosTest, CudaTryNoThrow) {
    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY(cudaSuccess));
    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY(cudaSuccess, std::runtime_error));
}

// Test RAPIDSMPF_CUDA_TRY macro with failed CUDA call (should throw)
TEST(ErrorMacrosTest, CudaTryThrow) {
    EXPECT_THROW(RAPIDSMPF_CUDA_TRY(cudaErrorInvalidValue), rapidsmpf::cuda_error);
    EXPECT_THROW(
        RAPIDSMPF_CUDA_TRY(cudaErrorInvalidValue, std::runtime_error), std::runtime_error
    );
}

// Test RAPIDSMPF_CUDA_TRY_ALLOC macro with successful CUDA call (should not throw)
TEST(ErrorMacrosTest, CudaTryAllocNoThrow) {
    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_ALLOC(cudaSuccess));
    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_ALLOC(cudaSuccess, 1024));
}

// Test that error messages contain expected information
TEST(ErrorMacrosTest, ErrorMessages) {
    // Test RAPIDSMPF_EXPECTS error message
    try {
        RAPIDSMPF_EXPECTS(false, "Test message");
        FAIL() << "Expected RAPIDSMPF_EXPECTS to throw an exception";
    } catch (const std::logic_error& e) {
        std::string error_message = e.what();
        EXPECT_TRUE(error_message.find("RAPIDSMPF failure at:") != std::string::npos);
        EXPECT_TRUE(error_message.find("Test message") != std::string::npos);
    }

    // Test RAPIDSMPF_FAIL error message
    try {
        RAPIDSMPF_FAIL("Test failure message");
        FAIL() << "Expected RAPIDSMPF_FAIL to throw an exception";
    } catch (const std::logic_error& e) {
        std::string error_message = e.what();
        EXPECT_TRUE(error_message.find("RAPIDSMPF failure at:") != std::string::npos);
        EXPECT_TRUE(error_message.find("Test failure message") != std::string::npos);
    }

    // Test RAPIDSMPF_CUDA_TRY error message
    try {
        RAPIDSMPF_CUDA_TRY(cudaErrorInvalidValue);
        FAIL() << "Expected RAPIDSMPF_CUDA_TRY to throw an exception";
    } catch (const rapidsmpf::cuda_error& e) {
        std::string error_message = e.what();
        EXPECT_TRUE(error_message.find("CUDA error at:") != std::string::npos);
        EXPECT_TRUE(error_message.find("invalid argument") != std::string::npos);
    }

    // Test RAPIDSMPF_CUDA_TRY_ALLOC error message (without bytes)
    try {
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaErrorInvalidValue);
        FAIL() << "Expected RAPIDSMPF_CUDA_TRY_ALLOC to throw an exception";
    } catch (const std::bad_alloc& e) {
        std::string error_message = e.what();
        EXPECT_TRUE(error_message.find("CUDA error at:") != std::string::npos);
        EXPECT_TRUE(error_message.find("invalid argument") != std::string::npos);
    }

    // Test RAPIDSMPF_CUDA_TRY_ALLOC error message (with bytes)
    try {
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaErrorInvalidValue, 1024);
        FAIL() << "Expected RAPIDSMPF_CUDA_TRY_ALLOC to throw an exception";
    } catch (const rapidsmpf::bad_alloc& e) {
        std::string error_message = e.what();
        EXPECT_TRUE(
            error_message.find("CUDA error (failed to allocate 1024 bytes)")
            != std::string::npos
        );
        EXPECT_TRUE(error_message.find("invalid argument") != std::string::npos);
    }

    // Test RAPIDSMPF_CUDA_TRY_ALLOC out_of_memory error message
    try {
        RAPIDSMPF_CUDA_TRY_ALLOC(cudaErrorMemoryAllocation, 2048);
        FAIL() << "Expected RAPIDSMPF_CUDA_TRY_ALLOC to throw an exception";
    } catch (const rapidsmpf::out_of_memory& e) {
        std::string error_message = e.what();
        EXPECT_TRUE(error_message.find("out_of_memory") != std::string::npos);
        EXPECT_TRUE(
            error_message.find("failed to allocate 2048 bytes") != std::string::npos
        );
        EXPECT_TRUE(error_message.find("out of memory") != std::string::npos);
    }
}

// Test fatal error functions (non-death tests for success cases)
TEST(ErrorMacrosTest, FatalFunctionsNoTerminate) {
    // RAPIDSMPF_EXPECTS_FATAL should not terminate when condition is true
    EXPECT_NO_THROW(RAPIDSMPF_EXPECTS_FATAL(true, "This should not terminate"));

    // RAPIDSMPF_CUDA_TRY_FATAL should not terminate on success
    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(cudaSuccess));
}

// Test fatal error functions (single consolidated death test)
TEST(ErrorMacrosDeathTest, FatalFunctions) {
    // Test RAPIDSMPF_EXPECTS_FATAL terminates on false condition
    EXPECT_DEATH(
        RAPIDSMPF_EXPECTS_FATAL(false, "Test expects fatal"),
        "RAPIDSMPF FATAL ERROR.*Test expects fatal"
    );

    // Test RAPIDSMPF_FATAL always terminates
    EXPECT_DEATH(
        RAPIDSMPF_FATAL("Test fatal message"), "RAPIDSMPF FATAL ERROR.*Test fatal"
    );

    // Test RAPIDSMPF_CUDA_TRY_FATAL terminates on CUDA error
    EXPECT_DEATH(
        RAPIDSMPF_CUDA_TRY_FATAL(cudaErrorInvalidValue),
        "RAPIDSMPF FATAL ERROR.*invalid argument"
    );
}

// Test actual CUDA operations with the macros
TEST(ErrorMacrosTest, ActualCudaOperations) {
    // Test successful memory allocation and free
    void* d_ptr = nullptr;
    constexpr size_t test_allocation_size = 1024;

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_ALLOC(
        cudaMalloc(&d_ptr, test_allocation_size), test_allocation_size
    ));
    ASSERT_NE(d_ptr, nullptr);

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY(cudaFree(d_ptr)));

    // Test successful CUDA operation
    std::array<int, 5> h_data = {1, 2, 3, 4, 5};
    int* d_data = nullptr;

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_ALLOC(
        cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(h_data)), sizeof(h_data)
    ));

    ASSERT_NE(d_data, nullptr);

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(d_data, h_data.data(), sizeof(h_data), cudaMemcpyHostToDevice)
    ));

    std::array<int, 5> h_result = {0};
    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY(
        cudaMemcpy(h_result.data(), d_data, sizeof(h_result), cudaMemcpyDeviceToHost)
    ));

    for (size_t i = 0; i < h_data.size(); ++i) {
        EXPECT_EQ(h_data[i], h_result[i]);
    }

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY(cudaFree(d_data)));
}

// Test actual CUDA operations with the fatal macros
TEST(ErrorMacrosTest, ActualCudaOperationsFatal) {
    // Test successful memory allocation and free with fatal macros
    void* d_ptr = nullptr;
    constexpr size_t test_allocation_size = 1024;

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(cudaMalloc(&d_ptr, test_allocation_size)));
    ASSERT_NE(d_ptr, nullptr);

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(cudaFree(d_ptr)));

    // Test successful CUDA memcpy operation
    int* d_data = nullptr;
    constexpr int test_value = 42;
    int h_result = 0;

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(
        cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(int))
    ));
    ASSERT_NE(d_data, nullptr);

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(
        cudaMemcpy(d_data, &test_value, sizeof(int), cudaMemcpyHostToDevice)
    ));

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(
        cudaMemcpy(&h_result, d_data, sizeof(int), cudaMemcpyDeviceToHost)
    ));

    EXPECT_EQ(test_value, h_result);

    EXPECT_NO_THROW(RAPIDSMPF_CUDA_TRY_FATAL(cudaFree(d_data)));
}
