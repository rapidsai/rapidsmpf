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
BackendType detect_backend() {
    // Check for rrun coordination first (explicit configuration takes priority).
    // If RAPIDSMPF_COORD_DIR is set, rrun is coordinating and we should use FILE backend.
    if (getenv_optional("RAPIDSMPF_COORD_DIR")) {
        return BackendType::FILE;
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
        return BackendType::SLURM;
    }
#endif

    // Default to file-based
    return BackendType::FILE;
}

/**
 * @brief Initialize context for FILE backend.
 */
Context file_backend_init() {
    Context ctx;
    ctx.type = BackendType::FILE;

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

    return ctx;
}

#ifdef RAPIDSMPF_HAVE_SLURM
/**
 * @brief Initialize context for SLURM backend.
 */
Context slurm_backend_init() {
    Context ctx;
    ctx.type = BackendType::SLURM;

    try {
        ctx.rank = get_rank();
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            "Could not determine rank for Slurm backend. "
            "Ensure you're running with 'srun --mpi=pmix'."
        );
    }

    try {
        ctx.nranks = get_nranks();
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(
            "Could not determine nranks for Slurm backend. "
            "Ensure you're running with 'srun --mpi=pmix'."
        );
    }

    if (!(ctx.rank >= 0 && ctx.rank < ctx.nranks)) {
        throw std::runtime_error(
            "Invalid rank: " + std::to_string(ctx.rank) + " must be in range [0, "
            + std::to_string(ctx.nranks) + ")"
        );
    }

    return ctx;
}
#endif
}  // namespace

Context init(BackendType type) {
    if (type == BackendType::AUTO) {
        type = detect_backend();
    }

    Context ctx;

    // Get rank and nranks based on backend, then create backend instance
    switch (type) {
    case BackendType::FILE:
        ctx = file_backend_init();
        ctx.backend = std::make_shared<detail::FileBackend>(ctx);
        break;
#ifdef RAPIDSMPF_HAVE_SLURM
    case BackendType::SLURM:
        ctx = slurm_backend_init();
        ctx.backend = std::make_shared<detail::SlurmBackend>(ctx);
        break;
#else
    case BackendType::SLURM:
        throw std::runtime_error(
            "SLURM backend requested but rapidsmpf was not built with PMIx support. "
            "Rebuild with RAPIDSMPF_ENABLE_SLURM=ON and ensure PMIx is available."
        );
#endif
    case BackendType::AUTO:
        // Should have been resolved above
        throw std::logic_error("BackendType::AUTO should have been resolved");
    }

    return ctx;
}

void broadcast(Context const& ctx, void* data, std::size_t size) {
    if (!ctx.backend) {
        throw std::runtime_error("Context not properly initialized - backend is null");
    }
    ctx.backend->broadcast(data, size);
}

void barrier(Context const& ctx) {
    if (!ctx.backend) {
        throw std::runtime_error("Context not properly initialized - backend is null");
    }
    ctx.backend->barrier();
}

void sync(Context const& ctx) {
    if (!ctx.backend) {
        throw std::runtime_error("Context not properly initialized - backend is null");
    }
    ctx.backend->sync();
}

void put(Context const& ctx, std::string const& key, std::string const& value) {
    if (!ctx.backend) {
        throw std::runtime_error("Context not properly initialized - backend is null");
    }
    ctx.backend->put(key, value);
}

std::string get(Context const& ctx, std::string const& key, Duration timeout) {
    if (!ctx.backend) {
        throw std::runtime_error("Context not properly initialized - backend is null");
    }
    return ctx.backend->get(key, timeout);
}

}  // namespace rapidsmpf::bootstrap
