/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/config.hpp>

#ifdef RAPIDSMPF_HAVE_SLURM

#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <rapidsmpf/bootstrap/slurm_backend.hpp>

// NOTE: Do not use RAPIDSMPF_EXPECTS or RAPIDSMPF_FAIL in this file.
// Using these macros introduces a CUDA dependency via rapidsmpf/error.hpp.
// Prefer throwing standard exceptions instead.

namespace rapidsmpf::bootstrap::detail {

namespace {

// PMIx initialization is process-global and must only happen once.
// Once initialized, PMIx stays active for the lifetime of the process.
// We track initialization state but do NOT finalize PMIx in the destructor,
// as multiple SlurmBackend instances may be created/destroyed during the
// bootstrap process. PMIx will be cleaned up when the process exits.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::mutex g_pmix_mutex;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool g_pmix_initialized = false;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
pmix_proc_t g_pmix_proc{};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::array<char, PMIX_MAX_NSLEN + 1> g_pmix_nspace{};

/**
 * @brief Convert PMIx status to string for error messages.
 *
 * @param status PMIx status code to convert.
 * @return Human-readable string describing the status.
 */
std::string pmix_error_string(pmix_status_t status) {
    return std::string{PMIx_Error_string(status)};
}

/**
 * @brief Check PMIx status and throw on error.
 *
 * @param status PMIx status code to check.
 * @param operation Description of the operation (used in error message).
 * @throws std::runtime_error if status is not PMIX_SUCCESS.
 */
void check_pmix_status(pmix_status_t status, std::string const& operation) {
    if (status != PMIX_SUCCESS) {
        throw std::runtime_error(operation + " failed: " + pmix_error_string(status));
    }
}

}  // namespace

SlurmBackend::SlurmBackend(Context ctx) : ctx_{std::move(ctx)} {
    std::lock_guard<std::mutex> lock{g_pmix_mutex};

    if (!g_pmix_initialized) {
        // First instance - initialize PMIx (will stay initialized for process lifetime)
        pmix_proc_t proc;
        pmix_status_t rc = PMIx_Init(&proc, nullptr, 0);
        if (rc != PMIX_SUCCESS) {
            throw std::runtime_error(
                "PMIx_Init failed: " + pmix_error_string(rc)
                + ". Ensure you're running under Slurm with --mpi=pmix"
            );
        }

        g_pmix_proc = proc;
        // Copy full nspace buffer (both are PMIX_MAX_NSLEN + 1 in size)
        static_assert(sizeof(proc.nspace) == PMIX_MAX_NSLEN + 1);
        std::memcpy(g_pmix_nspace.data(), proc.nspace, g_pmix_nspace.size());
        g_pmix_initialized = true;
    }

    pmix_initialized_ = true;

    // Copy global state to instance members
    proc_ = g_pmix_proc;
    nspace_ = g_pmix_nspace;

    // Verify rank matches what we expect (if context has a valid rank)
    // Note: For SLURM backend, ctx_.rank may be set from environment variables
    // before PMIx_Init, so we verify they match
    if (ctx_.rank >= 0 && std::cmp_not_equal(g_pmix_proc.rank, ctx_.rank)) {
        throw std::runtime_error(
            "PMIx rank (" + std::to_string(g_pmix_proc.rank)
            + ") doesn't match context rank (" + std::to_string(ctx_.rank) + ")"
        );
    }

    // Update context rank from PMIx if not already set
    if (ctx_.rank < 0) {
        ctx_.rank = static_cast<Rank>(g_pmix_proc.rank);
    }
}

SlurmBackend::~SlurmBackend() {
    // Intentionally do NOT call PMIx_Finalize here.
    // PMIx must stay initialized for the lifetime of the process because
    // multiple SlurmBackend instances may be created and destroyed during
    // bootstrap operations (put, barrier, get each create a new instance).
    //
    // TODO: Check whether it's safe to let PMIx clean itself up when the
    // process exits, and potentially come up with a better solution. Maybe
    // refcounting?
}

void SlurmBackend::put(std::string const& key, std::string const& value) {
    pmix_value_t pmix_value;
    PMIX_VALUE_CONSTRUCT(&pmix_value);
    pmix_value.type = PMIX_BYTE_OBJECT;
    pmix_value.data.bo.bytes = const_cast<char*>(value.data());
    pmix_value.data.bo.size = value.size();

    pmix_status_t rc = PMIx_Put(PMIX_GLOBAL, key.c_str(), &pmix_value);
    if (rc != PMIX_SUCCESS) {
        throw std::runtime_error(
            "PMIx_Put for key '" + key + "' failed: " + pmix_error_string(rc)
        );
    }

    // Commit to make the data available
    commit();
}

void SlurmBackend::commit() {
    pmix_status_t rc = PMIx_Commit();
    check_pmix_status(rc, "PMIx_Commit");
}

std::string SlurmBackend::get(std::string const& key, Duration timeout) {
    auto start = std::chrono::steady_clock::now();
    auto poll_interval = std::chrono::milliseconds{100};

    // Get from rank 0 specifically (since that's where the key is stored)
    // Using PMIX_RANK_WILDCARD doesn't seem to work reliably
    pmix_proc_t proc;
    PMIX_PROC_CONSTRUCT(&proc);
    std::memcpy(proc.nspace, nspace_.data(), nspace_.size());
    proc.rank = 0;  // Get from rank 0 specifically

    while (true) {
        pmix_value_t* val = nullptr;
        pmix_status_t rc = PMIx_Get(&proc, key.c_str(), nullptr, 0, &val);

        if (rc == PMIX_SUCCESS && val != nullptr) {
            std::string result;

            if (val->type == PMIX_BYTE_OBJECT) {
                result = std::string{
                    static_cast<char const*>(val->data.bo.bytes), val->data.bo.size
                };
            } else if (val->type == PMIX_STRING) {
                result = std::string{val->data.string};
            } else {
                PMIX_VALUE_RELEASE(val);
                throw std::runtime_error(
                    "Unexpected PMIx value type for key '" + key
                    + "': " + std::to_string(static_cast<int>(val->type))
                );
            }

            PMIX_VALUE_RELEASE(val);
            return result;
        }

        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            throw std::runtime_error(
                "Key '" + key + "' not available within "
                + std::to_string(
                    std::chrono::duration_cast<std::chrono::seconds>(timeout).count()
                )
                + "s timeout (last error: " + pmix_error_string(rc) + ")"
            );
        }

        // Sleep before retry
        std::this_thread::sleep_for(poll_interval);
    }
}

void SlurmBackend::barrier() {
    // Create proc array for all ranks (wildcard) in our namespace
    pmix_proc_t proc;
    PMIX_PROC_CONSTRUCT(&proc);
    std::memcpy(proc.nspace, nspace_.data(), nspace_.size());
    proc.rank = PMIX_RANK_WILDCARD;

    // Set up info to collect data during fence
    pmix_info_t info;
    bool collect = true;
    PMIX_INFO_CONSTRUCT(&info);
    PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &collect, PMIX_BOOL);

    // PMIx_Fence performs synchronization barrier and data exchange
    pmix_status_t rc = PMIx_Fence(&proc, 1, &info, 1);
    PMIX_INFO_DESTRUCT(&info);

    // Accept both SUCCESS and PARTIAL_SUCCESS for the fence
    // PARTIAL_SUCCESS can occur in some PMIx implementations when not all
    // processes have data to contribute, but the synchronization succeeded
    if (rc != PMIX_SUCCESS && rc != PMIX_ERR_PARTIAL_SUCCESS) {
        throw std::runtime_error("PMIx_Fence (barrier) failed: " + pmix_error_string(rc));
    }
}

void SlurmBackend::broadcast(void* data, std::size_t size, Rank root) {
    // Use unique key for each broadcast to avoid collisions
    std::string bcast_key =
        "bcast_" + std::to_string(root) + "_" + std::to_string(barrier_count_++);

    if (ctx_.rank == root) {
        // Root publishes data
        std::string bcast_data{static_cast<char const*>(data), size};
        put(bcast_key, bcast_data);
    }

    barrier();

    if (ctx_.rank != root) {
        // Non-root ranks retrieve data
        std::string bcast_data = get(bcast_key, std::chrono::seconds{30});
        if (bcast_data.size() != size) {
            throw std::runtime_error(
                "Broadcast size mismatch: expected " + std::to_string(size) + ", got "
                + std::to_string(bcast_data.size())
            );
        }
        std::memcpy(data, bcast_data.data(), size);
    }

    barrier();
}

}  // namespace rapidsmpf::bootstrap::detail

#endif  // RAPIDSMPF_HAVE_SLURM
