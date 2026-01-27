/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Main entry point for rrun topology binding tests.
 *
 * These tests validate rrun's topology-based binding functionality and must be
 * run with rrun, e.g.:
 *   rrun -n 1 gtests/rrun_tests
 *   rrun -n 4 gtests/rrun_tests --gtest_filter="*TopologyBinding*"
 */

#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
