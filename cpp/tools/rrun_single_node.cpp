/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "rrun_utils.hpp"

namespace rrun {

int execute_single_node_mode(Config& cfg) {
    if (cfg.verbose) {
        std::cout << "[rrun] Single-node mode: launching " << cfg.nranks << " ranks"
                  << std::endl;
    }

    return setup_launch_and_cleanup(cfg, 0, cfg.nranks, cfg.nranks);
}

}  // namespace rrun
