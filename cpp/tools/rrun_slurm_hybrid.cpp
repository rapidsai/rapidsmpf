/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#include <pmix.h>
#include <unistd.h>

#include <rapidsmpf/bootstrap/slurm_backend.hpp>

#include "rrun_utils.hpp"

namespace rrun {

namespace {

/**
 * @brief Write a value into the FileBackend key-value store.
 *
 * Mirrors the atomic write-then-rename pattern used by FileBackend::write_file
 * so that the children's `get()` call never sees partial content.
 *
 * @param coord_dir Local coordination directory (RRUN_COORD_DIR).
 * @param key       Key name (e.g. "ucxx_root_address").
 * @param value     Binary-safe value to store.
 */
void write_kv_for_children(
    std::string const& coord_dir, std::string const& key, std::string_view value
) {
    std::string kv_dir = coord_dir + "/kv";
    std::filesystem::create_directories(kv_dir);

    std::string path = kv_dir + "/" + key;
    std::string tmp_path = path + ".tmp." + std::to_string(getpid());

    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open temporary file: " + tmp_path);
    }
    ofs.write(value.data(), static_cast<std::streamsize>(value.size()));
    ofs.close();

    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::error_code rm_ec;
        std::filesystem::remove(tmp_path, rm_ec);
        throw std::runtime_error(
            "Failed to rename " + tmp_path + " to " + path + ": " + ec.message()
        );
    }
}

/**
 * @brief Relay the root UCXX address from rank 0 to all Slurm tasks.
 *
 * Called in a background thread after all ranks have been launched. The root
 * parent (SLURM_PROCID==0) polls the address file that rank 0 writes when it
 * initialises UCXX. Once available it publishes the address via PMIx so that
 * non-root parents can retrieve it. Every parent then writes the address into
 * its local FileBackend kv store so that the children can pick it up with
 * `get("ucxx_root_address")`.
 *
 * If rank 0 never writes the address (e.g. diagnostic tools that don't use
 * UCXX) the thread simply waits until @p stop becomes true (set by the caller
 * after all children have exited) and returns without doing anything.
 *
 * @param cfg           Configuration.
 * @param address_file  Path to the file rank 0 writes its address to.
 * @param stop          Atomic flag set by the main thread once children exit.
 */
void relay_root_address(
    Config const& cfg, std::string const& address_file, std::atomic<bool> const& stop
) {
    bool is_root_parent = (cfg.slurm->global_rank == 0);
    std::string encoded_address;

    if (is_root_parent) {
        // Poll for the address file from rank 0.
        while (!stop.load(std::memory_order_relaxed)) {
            if (std::filesystem::exists(address_file)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (stop.load(std::memory_order_relaxed)
            && !std::filesystem::exists(address_file))
        {
            // Children exited without rank 0 writing the address; nothing to relay.
            return;
        }

        std::ifstream addr_stream(address_file);
        if (!addr_stream) {
            std::cerr << "[rrun] Warning: failed to open root address file: "
                      << address_file << std::endl;
            return;
        }
        std::getline(addr_stream, encoded_address);
        addr_stream.close();
        std::filesystem::remove(address_file);

        if (encoded_address.empty()) {
            std::cerr << "[rrun] Warning: root address file was empty" << std::endl;
            return;
        }

        if (cfg.verbose) {
            std::cout << "[rrun] Got root address from rank 0 (hex-encoded, "
                      << encoded_address.size() << " chars)" << std::endl;
        }
    }

    // PMIx coordination between parents.
    int parent_rank = cfg.slurm->global_rank;
    int parent_nranks = cfg.slurm->ntasks;

    rapidsmpf::bootstrap::Context parent_ctx{
        parent_rank,
        parent_nranks,
        rapidsmpf::bootstrap::BackendType::SLURM,
        std::nullopt,
        nullptr
    };

    auto backend =
        std::make_shared<rapidsmpf::bootstrap::detail::SlurmBackend>(parent_ctx);

    if (is_root_parent) {
        if (cfg.verbose) {
            std::cout << "[rrun] Publishing root address via PMIx (hex-encoded, "
                      << encoded_address.size() << " chars)" << std::endl;
        }
        backend->put("rapidsmpf_root_address", encoded_address);
    }

    backend->sync();

    if (!is_root_parent) {
        encoded_address =
            backend->get("rapidsmpf_root_address", std::chrono::seconds{30});
        if (cfg.verbose) {
            std::cout << "[rrun] Retrieved root address via PMIx (hex-encoded, "
                      << encoded_address.size() << " chars)" << std::endl;
        }
    }

    // Write into the local FileBackend kv store so children can get() it.
    // Rank 0's UCXX bootstrap publishes the raw binary address via put(), and
    // non-root ranks retrieve it via get(). The FileBackend kv entry must
    // therefore contain the raw address, not the hex-encoded one.
    //
    // Hex encoding is only used for the address file and PMIx transport (which
    // may not be binary-safe). Decode before writing to the kv store.

    // Inline hex decode (same as in ucxx.cpp).
    auto hex_decode = [](std::string_view const& input) -> std::string {
        std::string result;
        result.reserve(input.size() / 2);
        for (std::size_t i = 0; i + 1 < input.size(); i += 2) {
            auto high = static_cast<unsigned char>(
                (input[i] >= 'a') ? (input[i] - 'a' + 10) : (input[i] - '0')
            );
            auto low = static_cast<unsigned char>(
                (input[i + 1] >= 'a') ? (input[i + 1] - 'a' + 10) : (input[i + 1] - '0')
            );
            result.push_back(static_cast<char>((high << 4) | low));
        }
        return result;
    };

    std::string raw_address = hex_decode(encoded_address);
    write_kv_for_children(cfg.coord_dir, "ucxx_root_address", raw_address);

    if (cfg.verbose) {
        std::cout << "[rrun] Wrote root address to local kv store for children"
                  << std::endl;
    }
}

}  // namespace

int execute_slurm_hybrid_mode(Config& cfg) {
    if (!cfg.slurm || cfg.slurm->job_id < 0 || cfg.slurm->ntasks <= 0
        || cfg.slurm->global_rank < 0)
    {
        throw std::runtime_error(
            "SLURM_JOB_ID, SLURM_NTASKS and SLURM_PROCID must be set for Slurm hybrid "
            "mode. Ensure you are running under srun with --ntasks (or equivalent)."
        );
    }

    int total_ranks = cfg.slurm->ntasks * cfg.nranks;
    int rank_offset = cfg.slurm->global_rank * cfg.nranks;
    std::string coord_hint = "slurm_" + std::to_string(cfg.slurm->job_id);

    if (cfg.verbose) {
        std::cout << "[rrun] Slurm hybrid mode: task " << cfg.slurm->global_rank
                  << " launching " << cfg.nranks << " ranks per task" << std::endl;
    }

    // Slurm hybrid mode uses the FileBackend for intra-task coordination.
    // The SocketServer is currently task-local only.
    cfg.file_backend = true;

    // Set up coordination directory (created by setup_launch_and_cleanup).
    if (cfg.coord_dir.empty()) {
        cfg.coord_dir = "/tmp/rrun_slurm_" + std::to_string(cfg.slurm->job_id);
    }

    // Set RRUN_ROOT_ADDRESS_FILE for rank 0 (only on the root parent).
    // Rank 0's UCXX bootstrap writes its listener address here in addition
    // to publishing it via put().
    std::string address_file =
        "/tmp/rapidsmpf_root_address_" + std::to_string(cfg.slurm->job_id);
    bool is_root_parent = (cfg.slurm->global_rank == 0);
    if (is_root_parent) {
        setenv("RRUN_ROOT_ADDRESS_FILE", address_file.c_str(), 1);
    }

    // Launch all ranks (including rank 0) simultaneously.
    // The ranks use the FileBackend for UCXX bootstrap coordination (rank 0
    // calls put(), non-root ranks call get()). The relay thread below will
    // populate the local kv store with the root address once it becomes
    // available.
    std::atomic<bool> relay_stop{false};
    std::thread relay_thread(
        relay_root_address, std::cref(cfg), std::cref(address_file), std::cref(relay_stop)
    );

    int exit_status =
        setup_launch_and_cleanup(cfg, rank_offset, cfg.nranks, total_ranks, coord_hint);

    // Children have exited; tell the relay thread to stop if it hasn't already.
    relay_stop.store(true, std::memory_order_relaxed);

    unsetenv("RRUN_ROOT_ADDRESS_FILE");

    if (relay_thread.joinable()) {
        relay_thread.join();
    }

    return exit_status;
}

}  // namespace rrun
