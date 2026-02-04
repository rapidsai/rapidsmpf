/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/file_backend.hpp>
#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_SLURM
#include <rapidsmpf/bootstrap/slurm_backend.hpp>
#endif

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rapidsmpf::bootstrap {
namespace {

/**
 * @brief Detect backend from environment variables.
 */
Backend detect_backend() {
    // Check for rrun coordination first (explicit configuration takes priority)
    // If RAPIDSMPF_COORD_DIR or RAPIDSMPF_ROOT_ADDRESS is set, rrun is coordinating
    // and we should use FILE backend (with or without pre-coordinated address)
    if (getenv_optional("RAPIDSMPF_COORD_DIR")
        || getenv_optional("RAPIDSMPF_ROOT_ADDRESS"))
    {
        return Backend::FILE;
    }

#ifdef RAPIDSMPF_HAVE_SLURM
    // Check for Slurm-specific environment variables ONLY if rrun is NOT coordinating.
    // This allows direct use of Slurm/PMIx backend when NOT launched via rrun.
    // Note: We don't check PMIX_NAMESPACE alone because OpenMPI also uses PMIx
    // internally and sets PMIX_NAMESPACE when launched with mpirun.
    // SLURM_JOB_ID + SLURM_PROCID is specific to Slurm srun tasks.
    //
    // Important: This path should only be taken by Slurm parent processes that are
    // NOT launched by rrun. Child processes launched by rrun will have RAPIDSMPF_*
    // variables set and will use FILE backend above.
    if (is_running_with_slurm()) {
        return Backend::SLURM;
    }
#endif

    // Default to file-based
    return Backend::FILE;
}
}  // namespace

Context init(Backend backend) {
    Context ctx;
    ctx.backend = (backend == Backend::AUTO) ? detect_backend() : backend;

    // Get rank and nranks based on backend
    switch (ctx.backend) {
    case Backend::FILE:
        {
            // Require explicit RAPIDSMPF_RANK and RAPIDSMPF_NRANKS
            auto rank_opt = getenv_int("RAPIDSMPF_RANK");
            auto nranks_opt = getenv_int("RAPIDSMPF_NRANKS");
            auto coord_dir_opt = getenv_optional("RAPIDSMPF_COORD_DIR");

            if (!rank_opt.has_value()) {
                throw std::runtime_error(
                    "RAPIDSMPF_RANK environment variable not set. "
                    "Set it or use a launcher like 'rrun'."
                );
            }

            if (!nranks_opt.has_value()) {
                throw std::runtime_error(
                    "RAPIDSMPF_NRANKS environment variable not set. "
                    "Set it or use a launcher like 'rrun'."
                );
            }

            if (!coord_dir_opt.has_value()) {
                throw std::runtime_error(
                    "RAPIDSMPF_COORD_DIR environment variable not set. "
                    "Set it or use a launcher like 'rrun'."
                );
            }

            ctx.rank = static_cast<Rank>(*rank_opt);
            ctx.nranks = static_cast<Rank>(*nranks_opt);
            ctx.coord_dir = *coord_dir_opt;

            if (!(ctx.rank >= 0 && ctx.rank < ctx.nranks)) {
                throw std::runtime_error(
                    "Invalid rank: RAPIDSMPF_RANK=" + std::to_string(ctx.rank)
                    + " must be in range [0, " + std::to_string(ctx.nranks) + ")"
                );
            }
            break;
        }
    case Backend::SLURM:
        {
#ifdef RAPIDSMPF_HAVE_SLURM
            // For SLURM backend, we can get rank/nranks from multiple sources:
            // 1. Explicit RAPIDSMPF_* variables (override)
            // 2. PMIx environment variables (set by pmix-enabled srun)
            // 3. Slurm environment variables (fallback)
            auto rank_opt = getenv_int("RAPIDSMPF_RANK");
            if (!rank_opt) {
                rank_opt = getenv_int("PMIX_RANK");
            }
            if (!rank_opt) {
                rank_opt = getenv_int("SLURM_PROCID");
            }

            auto nranks_opt = getenv_int("RAPIDSMPF_NRANKS");
            if (!nranks_opt) {
                nranks_opt = getenv_int("SLURM_NPROCS");
            }
            if (!nranks_opt) {
                nranks_opt = getenv_int("SLURM_NTASKS");
            }

            if (!rank_opt.has_value()) {
                throw std::runtime_error(
                    "Could not determine rank for SLURM backend. "
                    "Ensure you're running with 'srun --mpi=pmix' or set RAPIDSMPF_RANK."
                );
            }

            if (!nranks_opt.has_value()) {
                throw std::runtime_error(
                    "Could not determine nranks for SLURM backend. "
                    "Ensure you're running with 'srun --mpi=pmix' or set "
                    "RAPIDSMPF_NRANKS."
                );
            }

            ctx.rank = static_cast<Rank>(*rank_opt);
            ctx.nranks = static_cast<Rank>(*nranks_opt);

            if (!(ctx.rank >= 0 && ctx.rank < ctx.nranks)) {
                throw std::runtime_error(
                    "Invalid rank: " + std::to_string(ctx.rank) + " must be in range [0, "
                    + std::to_string(ctx.nranks) + ")"
                );
            }
            break;
#else
            throw std::runtime_error(
                "SLURM backend requested but rapidsmpf was not built with PMIx support. "
                "Rebuild with RAPIDSMPF_ENABLE_SLURM=ON and ensure PMIx is available."
            );
#endif
        }
    case Backend::AUTO:
        {
            // Should have been resolved above
            throw std::logic_error("Backend::AUTO should have been resolved");
        }
    }
    return ctx;
}

void broadcast(Context const& ctx, void* data, std::size_t size, Rank root) {
    switch (ctx.backend) {
    case Backend::FILE:
        {
            detail::FileBackend backend{ctx};
            backend.broadcast(data, size, root);
            break;
        }
#ifdef RAPIDSMPF_HAVE_SLURM
    case Backend::SLURM:
        {
            detail::SlurmBackend backend{ctx};
            backend.broadcast(data, size, root);
            break;
        }
#endif
    default:
        throw std::runtime_error("broadcast not implemented for this backend");
    }
}

void barrier(Context const& ctx) {
    switch (ctx.backend) {
    case Backend::FILE:
        {
            detail::FileBackend backend{ctx};
            backend.barrier();
            break;
        }
#ifdef RAPIDSMPF_HAVE_SLURM
    case Backend::SLURM:
        {
            detail::SlurmBackend backend{ctx};
            backend.barrier();
            break;
        }
#endif
    default:
        throw std::runtime_error("barrier not implemented for this backend");
    }
}

void sync(Context const& ctx) {
    switch (ctx.backend) {
    case Backend::FILE:
        {
            detail::FileBackend backend{ctx};
            backend.sync();
            break;
        }
#ifdef RAPIDSMPF_HAVE_SLURM
    case Backend::SLURM:
        {
            detail::SlurmBackend backend{ctx};
            backend.sync();
            break;
        }
#endif
    default:
        throw std::runtime_error("sync not implemented for this backend");
    }
}

void put(Context const& ctx, std::string const& key, std::string const& value) {
    switch (ctx.backend) {
    case Backend::FILE:
        {
            detail::FileBackend backend{ctx};
            backend.put(key, value);
            break;
        }
#ifdef RAPIDSMPF_HAVE_SLURM
    case Backend::SLURM:
        {
            detail::SlurmBackend backend{ctx};
            backend.put(key, value);
            break;
        }
#endif
    default:
        throw std::runtime_error("put not implemented for this backend");
    }
}

std::string get(Context const& ctx, std::string const& key, Duration timeout) {
    switch (ctx.backend) {
    case Backend::FILE:
        {
            detail::FileBackend backend{ctx};
            return backend.get(key, timeout);
        }
#ifdef RAPIDSMPF_HAVE_SLURM
    case Backend::SLURM:
        {
            detail::SlurmBackend backend{ctx};
            return backend.get(key, timeout);
        }
#endif
    default:
        throw std::runtime_error("get not implemented for this backend");
    }
}

}  // namespace rapidsmpf::bootstrap
