/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Process launcher for multi-GPU applications (single-node).
 *
 * rrun is a lightweight alternative to mpirun that:
 * - Launches multiple processes locally without requiring MPI
 * - Automatically assigns GPUs to ranks
 * - Provides file-based coordination for inter-process synchronization
 * - Tags process output with rank numbers (--tag-output feature)
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "rrun_config.hpp"
#include "rrun_utils.hpp"

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rrun {

// Mode execution functions defined in separate translation units.
int execute_single_node_mode(Config& cfg);
[[noreturn]] void execute_slurm_passthrough_mode(Config const& cfg);
#ifdef RAPIDSMPF_HAVE_SLURM
int execute_slurm_hybrid_mode(Config& cfg);
#endif

namespace {

/**
 * @brief Print usage information.
 */
void print_usage(std::string_view prog_name) {
    std::cout
        << "rrun - RapidsMPF Process Launcher\n\n"
        << "Usage: " << prog_name << " [options] <application> [app_args...]\n\n"
        << "Single-Node Options:\n"
        << "  -n <nranks>        Number of ranks to launch (required in single-node "
        << "                     mode)\n"
        << "  -g <gpu_list>      Comma-separated list of GPU IDs (e.g., 0,1,2,3)\n"
        << "                     If not specified, auto-detect available GPUs\n\n"
        << "Slurm Options:\n"
        << "  --slurm            Run in Slurm mode (auto-detected when SLURM_JOB_ID is "
        << "                     set)\n"
        << "                     Two sub-modes:\n"
        << "                     1. Passthrough (no -n): Apply topology bindings and\n"
        << "                     executes application\n"
        << "                     2. Hybrid (with -n): Launch N ranks per Slurm task.\n"
        << "                     In hybrid mode, each Slurm task launches multiple\n"
        << "                     ranks with coordinated global rank numbering.\n"
        << "                     Topology bindings are applied to each rank.\n\n"
        << "Common Options:\n"
        << "  -d <coord_dir>     Coordination directory (default: /tmp/rrun_<random>)\n"
        << "                     Not applicable in Slurm mode\n"
        << "  --tag-output       Tag stdout and stderr with rank number\n"
        << "                     Not applicable in Slurm mode\n"
        << "  --bind-to <type>   Bind to topology resources (default: all)\n"
        << "                     Can be specified multiple times\n"
        << "                     Options: cpu, memory, network, all, none\n"
        << "                     Examples: --bind-to cpu --bind-to network\n"
        << "                               --bind-to none (disable all bindings)\n"
        << "  -x, --set-env <VAR=val>\n"
        << "                     Set environment variable for all ranks\n"
        << "                     Can be specified multiple times\n"
        << "  -v                 Verbose output\n"
        << "  --no-cleanup       Don't cleanup coordination directory on exit\n"
        << "  --file-backend     Use file-based coordination instead of the default\n"
        << "                     socket-based backend. Implies -d if not set.\n"
        << "  -h, --help         Display this help message\n\n"
        << "Environment Variables:\n"
        << "  CUDA_VISIBLE_DEVICES is set for each rank based on GPU assignment\n"
        << "  Additional environment variables can be passed with -x/--set-env\n\n"
        << "Single-Node Examples:\n"
        << "  # Launch 2 ranks with auto-detected GPUs:\n"
        << "  rrun -n 2 ./bench_comm -C ucxx -O all-to-all\n\n"
        << "  # Launch 4 ranks on specific GPUs:\n"
        << "  rrun -n 4 -g 0,1,2,3 ./bench_comm -C ucxx\n\n"
        << "  # Launch with custom environment variables:\n"
        << "  rrun -n 2 -x UCX_TLS=cuda_copy,cuda_ipc,rc,tcp -x MY_VAR=value "
           "./bench_comm\n\n"
        << "Slurm Examples:\n"
        << "  # Passthrough: multiple (4) tasks per node, one task per GPU, two nodes.\n"
        << "  srun --mpi=pmix --nodes=2 --ntasks-per-node=4 --cpus-per-task=36 \\\n"
        << "      --gpus-per-task=1 --gres=gpu:4 \\\n"
        << "      rrun ./benchmarks/bench_shuffle -C ucxx\n\n"
        << "  # Hybrid mode: one task per node, 4 GPUs per task, two nodes.\n"
        << "  srun --mpi=pmix --nodes=2 --ntasks-per-node=1 --cpus-per-task=144 \\\n"
        << "      --gpus-per-task=4 --gres=gpu:4 \\\n"
        << "      rrun -n 4 ./benchmarks/bench_shuffle -C ucxx\n\n"
        << std::endl;
}

/**
 * @brief Parse command-line arguments.
 */
Config parse_args(int argc, char* argv[]) {
    Config cfg;
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-n") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -n");
            }
            cfg.nranks = std::stoi(argv[++i]);
            if (cfg.nranks <= 0) {
                throw std::runtime_error(
                    "Invalid number of ranks: " + std::to_string(cfg.nranks)
                );
            }
        } else if (arg == "-g") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -g");
            }
            cfg.gpus = parse_gpu_list(argv[++i]);
        } else if (arg == "--tag-output") {
            cfg.tag_output = true;
        } else if (arg == "--bind-to") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for --bind-to");
            }
            std::string bind_type = argv[++i];
            if (bind_type == "none") {
                if (cfg.bind_state == BindToState::Specific
                    || cfg.bind_state == BindToState::All)
                {
                    throw std::runtime_error(
                        "--bind-to none cannot be combined with other --bind-to options"
                    );
                }
                cfg.bind_state = BindToState::None;
                cfg.bind_cpu = false;
                cfg.bind_memory = false;
                cfg.bind_network = false;
            } else if (bind_type == "all") {
                if (cfg.bind_state == BindToState::Specific
                    || cfg.bind_state == BindToState::None)
                {
                    throw std::runtime_error(
                        "--bind-to all cannot be combined with other --bind-to options"
                    );
                }
                cfg.bind_state = BindToState::All;
                cfg.bind_cpu = true;
                cfg.bind_memory = true;
                cfg.bind_network = true;
            } else {
                if (cfg.bind_state == BindToState::None
                    || cfg.bind_state == BindToState::All)
                {
                    throw std::runtime_error(
                        "--bind-to " + bind_type
                        + " cannot be combined with --bind-to none or --bind-to all"
                    );
                }
                cfg.bind_state = BindToState::Specific;
                if (bind_type == "cpu") {
                    cfg.bind_cpu = true;
                } else if (bind_type == "memory") {
                    cfg.bind_memory = true;
                } else if (bind_type == "network") {
                    cfg.bind_network = true;
                } else {
                    throw std::runtime_error(
                        "Invalid --bind-to option: " + bind_type
                        + ". Valid options: cpu, memory, network, all, none"
                    );
                }
            }
        } else if (arg == "-d") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -d");
            }
            cfg.coord_dir = argv[++i];
        } else if (arg == "-x" || arg == "--set-env") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing argument for -x/--set-env");
            }
            std::string env_spec = argv[++i];
            auto eq_pos = env_spec.find('=');
            if (eq_pos == std::string::npos) {
                throw std::runtime_error(
                    "Invalid environment variable format: " + env_spec
                    + ". Expected VAR=value"
                );
            }
            std::string var_name = env_spec.substr(0, eq_pos);
            std::string var_value = env_spec.substr(eq_pos + 1);
            if (var_name.empty()) {
                throw std::runtime_error("Empty environment variable name");
            }
            cfg.env_vars[var_name] = var_value;
        } else if (arg == "-v") {
            cfg.verbose = true;
        } else if (arg == "--no-cleanup") {
            cfg.cleanup = false;
        } else if (arg == "--file-backend") {
            cfg.file_backend = true;
        } else if (arg == "--slurm") {
            cfg.slurm_mode = true;
        } else if (arg == "--") {
            // Everything after -- is the application and its arguments
            if (i + 1 < argc) {
                cfg.app_binary = argv[i + 1];
                for (int j = i + 2; j < argc; ++j) {
                    cfg.app_args.push_back(argv[j]);
                }
            }
            break;
        } else if (arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        } else {
            // First non-option argument is the application binary
            cfg.app_binary = arg;
            // Rest are application arguments
            for (int j = i + 1; j < argc; ++j) {
                cfg.app_args.push_back(argv[j]);
            }
            break;
        }
        ++i;
    }

    // Validate configuration
    if (cfg.app_binary.empty()) {
        throw std::runtime_error("Missing application binary");
    }

    // Auto-detect Slurm mode if not explicitly specified
    if (!cfg.slurm_mode) {
        cfg.slurm_mode = detect_slurm_environment(cfg);
    } else {
        // --slurm was specified, populate Slurm info
        if (!detect_slurm_environment(cfg)) {
            throw std::runtime_error(
                "--slurm specified but required Slurm environment variables "
                "(SLURM_JOB_ID, SLURM_LOCALID) are not set. "
                "Ensure you're running under srun."
            );
        }
    }

    if (cfg.slurm_mode) {
        // Slurm mode validation
        if (!cfg.slurm || cfg.slurm->local_id < 0) {
            throw std::runtime_error(
                "SLURM_LOCALID environment variable not set or invalid"
            );
        }

        // In Slurm mode:
        // - If -n is specified: launch N ranks per Slurm task (hybrid mode)
        // - If -n is not specified: just apply bindings and exec (passthrough mode,
        //                           one rank per task)
        if (cfg.nranks <= 0) {
            cfg.nranks = 1;
        }
    } else {
        // Single-node mode validation
        if (cfg.nranks <= 0) {
            throw std::runtime_error(
                "Number of ranks (-n) must be specified and positive"
            );
        }
    }

    // Auto-detect GPUs if not specified
    if (cfg.gpus.empty()) {
        cfg.gpus = detect_gpus();
        if (cfg.gpus.empty()) {
            std::cerr << "[rrun] Warning: No GPUs detected. CUDA_VISIBLE_DEVICES will "
                         "not be set."
                      << std::endl;
        }
    }

    // Validate GPU count vs rank count (only warn in single-node mode)
    if (!cfg.slurm_mode && !cfg.gpus.empty()
        && cfg.nranks > static_cast<int>(cfg.gpus.size()))
    {
        std::cerr << "[rrun] Warning: Number of ranks (" << cfg.nranks
                  << ") exceeds number of GPUs (" << cfg.gpus.size()
                  << "). Multiple ranks will share GPUs." << std::endl;
    }

    // Generate coordination directory if not specified (not needed in Slurm mode)
    if (cfg.coord_dir.empty() && !cfg.slurm_mode) {
        cfg.coord_dir = "/tmp/rrun_" + generate_session_id();
    }

    // Default to "all" if --bind-to was not explicitly specified
    if (cfg.bind_state == BindToState::NotSpecified) {
        cfg.bind_cpu = true;
        cfg.bind_memory = true;
        cfg.bind_network = true;
    }

    return cfg;
}

}  // namespace
}  // namespace rrun

int main(int argc, char* argv[]) {
    try {
        // Parse arguments
        rrun::Config cfg = rrun::parse_args(argc, argv);

        if (cfg.verbose) {
            std::cout << "rrun configuration:\n";
            std::cout << "  Mode:          " << (cfg.slurm_mode ? "Slurm" : "Single-node")
                      << "\n"
                      << "  GPUs:          ";
            if (cfg.gpus.empty()) {
                std::cout << "(none)\n";
            } else {
                for (std::size_t i = 0; i < cfg.gpus.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << cfg.gpus[i];
                }
                std::cout << "\n";
            }
            if (cfg.slurm_mode) {
                std::cout << "  Slurm Local ID: " << cfg.slurm->local_id << "\n"
                          << "  Slurm Rank:     " << cfg.slurm->global_rank << "\n"
                          << "  Slurm NTasks:   " << cfg.slurm->ntasks << "\n";
            } else {
                if (cfg.tag_output) {
                    std::cout << "  Tag Output:    Yes\n";
                }
                std::cout << "  Ranks:         " << cfg.nranks << "\n"
                          << "  Backend:       " << (cfg.file_backend ? "file" : "socket")
                          << "\n";
                if (cfg.file_backend) {
                    std::cout << "  Coord Dir:     " << cfg.coord_dir << "\n"
                              << "  Cleanup:       " << (cfg.cleanup ? "yes" : "no")
                              << "\n";
                }
            }
            std::cout << "  Application:   " << cfg.app_binary << "\n";
            std::vector<std::string> bind_types;
            if (cfg.bind_cpu)
                bind_types.push_back("cpu");
            if (cfg.bind_memory)
                bind_types.push_back("memory");
            if (cfg.bind_network)
                bind_types.push_back("network");
            if (bind_types.empty()) {
                std::cout << "  Bind To:       none\n";
            } else {
                std::cout << "  Bind To:       ";
                for (std::size_t i = 0; i < bind_types.size(); ++i) {
                    if (i > 0)
                        std::cout << ", ";
                    std::cout << bind_types[i];
                }
                std::cout << "\n";
            }
            if (!cfg.env_vars.empty()) {
                std::cout << "  Env Vars:      ";
                bool first = true;
                for (auto const& env_pair : cfg.env_vars) {
                    if (!first)
                        std::cout << "                 ";
                    std::cout << env_pair.first << "=" << env_pair.second << "\n";
                    first = false;
                }
            }
            std::cout << std::endl;
        }

        if (cfg.slurm_mode) {
            if (cfg.nranks == 1) {
                // Slurm passthrough mode: single rank per task, no forking
                rrun::execute_slurm_passthrough_mode(cfg);
            }
            // Slurm hybrid mode: multiple ranks per task with PMIx coordination
#ifdef RAPIDSMPF_HAVE_SLURM
            return rrun::execute_slurm_hybrid_mode(cfg);
#else
            std::cerr << "[rrun] Error: Slurm hybrid mode requires PMIx support but "
                      << "rapidsmpf was not built with PMIx." << std::endl;
            std::cerr << "[rrun] Rebuild with -DBUILD_SLURM_SUPPORT=ON or use "
                         "passthrough mode "
                      << "(without -n flag)." << std::endl;
            return 1;
#endif
        } else {
            // Single-node mode
            return rrun::execute_single_node_mode(cfg);
        }

    } catch (std::exception const& e) {
        std::cerr << "[rrun] Error: " << e.what() << std::endl;
        std::cerr << "[rrun] Run with -h or --help for usage information." << std::endl;
        return 1;
    }
}
