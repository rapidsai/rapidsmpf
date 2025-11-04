/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <rapidsmpf/bootstrap/bootstrap.hpp>
#include <rapidsmpf/bootstrap/file_backend.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::bootstrap {
namespace {

/**
 * @brief Get environment variable as string.
 */
std::optional<std::string> getenv_optional(std::string_view name) {
    // std::getenv requires a null-terminated string; construct a std::string
    // to ensure this even when called with a non-literal std::string_view.
    char const* value = std::getenv(std::string{name}.c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string{value};
}

/**
 * @brief Parse integer from environment variable.
 */
std::optional<int> getenv_int(std::string_view name) {
    auto value = getenv_optional(name);
    if (!value) {
        return std::nullopt;
    }
    try {
        return std::stoi(*value);
    } catch (...) {
        throw std::runtime_error(
            std::string{"Failed to parse integer from environment variable "}
            + std::string{name} + ": " + *value
        );
    }
}

/**
 * @brief Detect backend from environment variables.
 */
Backend detect_backend() {
    // Check for file-based coordination
    if (getenv_optional("RAPIDSMPF_COORD_DIR")) {
        return Backend::FILE;
    }

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

            RAPIDSMPF_EXPECTS(
                rank_opt.has_value(),
                "RAPIDSMPF_RANK environment variable not set. "
                "Set it or use a launcher like 'rrun'."
            );

            RAPIDSMPF_EXPECTS(
                nranks_opt.has_value(),
                "RAPIDSMPF_NRANKS environment variable not set. "
                "Set it or use a launcher like 'rrun'."
            );

            RAPIDSMPF_EXPECTS(
                coord_dir_opt.has_value(),
                "RAPIDSMPF_COORD_DIR environment variable not set. "
                "Set it or use a launcher like 'rrun'."
            );

            ctx.rank = static_cast<Rank>(*rank_opt);
            ctx.nranks = static_cast<Rank>(*nranks_opt);
            ctx.coord_dir = *coord_dir_opt;

            RAPIDSMPF_EXPECTS(
                ctx.rank >= 0 && ctx.rank < ctx.nranks,
                "Invalid rank: RAPIDSMPF_RANK=" + std::to_string(ctx.rank)
                    + " must be in range [0, " + std::to_string(ctx.nranks) + ")"
            );
            break;
        }
    case Backend::AUTO:
        {
            // Should have been resolved above
            RAPIDSMPF_FAIL("Backend::AUTO should have been resolved", std::logic_error);
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
    default:
        RAPIDSMPF_FAIL("broadcast not implemented for this backend", std::runtime_error);
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
    default:
        RAPIDSMPF_FAIL("barrier not implemented for this backend", std::runtime_error);
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
    default:
        RAPIDSMPF_FAIL("put not implemented for this backend", std::runtime_error);
    }
}

std::string get(Context const& ctx, std::string const& key, Duration timeout) {
    switch (ctx.backend) {
    case Backend::FILE:
        {
            detail::FileBackend backend{ctx};
            return backend.get(key, timeout);
        }
    default:
        RAPIDSMPF_FAIL("get not implemented for this backend", std::runtime_error);
    }
}

}  // namespace rapidsmpf::bootstrap
