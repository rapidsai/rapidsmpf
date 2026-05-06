/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include <cucascade/memory/topology_discovery.hpp>

#include <rrun/rrun.hpp>

#include "rrun_utils.hpp"

namespace rrun {

[[noreturn]] void execute_slurm_passthrough_mode(Config const& cfg) {
    if (!cfg.slurm || cfg.slurm->ntasks <= 0) {
        throw std::runtime_error(
            "SLURM_NTASKS must be set for Slurm passthrough mode. "
            "Ensure you are running under srun with --ntasks (or equivalent)."
        );
    }

    if (cfg.verbose) {
        std::cout << "[rrun] Slurm passthrough mode: applying bindings and exec'ing"
                  << std::endl;
    }

    // Discover topology BEFORE narrowing CUDA_VISIBLE_DEVICES.
    cucascade::memory::topology_discovery discovery;
    bool const have_topology = discovery.discover();

    // Set rrun coordination environment variables so the application knows
    // it's being launched by rrun and should use bootstrap mode
    setenv("RRUN_RANK", std::to_string(cfg.slurm->global_rank).c_str(), 1);
    setenv("RRUN_NRANKS", std::to_string(cfg.slurm->ntasks).c_str(), 1);

    // Determine GPU for this Slurm task
    int gpu_id = -1;
    if (!cfg.gpus.empty()) {
        gpu_id =
            cfg.gpus[static_cast<std::size_t>(cfg.slurm->local_id) % cfg.gpus.size()];
        setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu_id).c_str(), 1);

        if (cfg.verbose) {
            std::cout << "[rrun] Slurm task (passthrough) local_id="
                      << cfg.slurm->local_id << " assigned to GPU " << gpu_id
                      << std::endl;
        }
    }

    // Set custom environment variables
    for (auto const& env_pair : cfg.env_vars) {
        setenv(env_pair.first.c_str(), env_pair.second.c_str(), 1);
    }

    if (have_topology) {
        try {
            rapidsmpf::rrun::bind(
                discovery.get_topology(),
                gpu_id >= 0
                    ? std::optional<unsigned int>(static_cast<unsigned int>(gpu_id))
                    : std::nullopt,
                {.cpu = cfg.bind_cpu,
                 .memory = cfg.bind_memory,
                 .network = cfg.bind_network}
            );
        } catch (std::exception const& e) {
            std::cerr << "[rrun] Warning: " << e.what() << std::endl;
        }
    } else if (cfg.verbose) {
        std::cerr << "[rrun] Warning: topology discovery failed; "
                  << "resource binding skipped." << std::endl;
    }

    exec_application(cfg);
}

}  // namespace rrun
