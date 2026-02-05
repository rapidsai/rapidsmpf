/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <string>

#include <rapidsmpf/bootstrap/backend.hpp>
#include <rapidsmpf/bootstrap/bootstrap.hpp>

namespace rapidsmpf::bootstrap::detail {

/**
 * @brief File-based coordination backend implementation.
 *
 * This class implements coordination using a shared directory on the filesystem.
 * It creates lock files and data files to coordinate between ranks.
 *
 * Directory structure:
 * ```
 * <coord_dir>/
 *   ├── rank_<N>_alive        # Created by each rank to signal presence
 *   ├── kv/
 *   │   ├── <key1>            # Key-value pairs
 *   │   └── <key2>
 *   └── barriers/
 *       └── barrier_<N>       # Barrier synchronization
 * ```
 */
class FileBackend : public Backend {
  public:
    /**
     * @brief Construct a file backend.
     *
     * @param ctx Bootstrap context containing rank and coordination directory.
     */
    explicit FileBackend(Context ctx);

    ~FileBackend() override;

    /**
     * @copydoc Backend::put
     */
    void put(std::string const& key, std::string const& value) override;

    /**
     * @copydoc Backend::get
     */
    std::string get(std::string const& key, Duration timeout) override;

    /**
     * @copydoc Backend::barrier
     */
    void barrier() override;

    /**
     * @copydoc Backend::sync
     */
    void sync() override;

    /**
     * @copydoc Backend::broadcast
     */
    void broadcast(void* data, std::size_t size, Rank root) override;

  private:
    Context ctx_;
    std::string coord_dir_;
    std::string kv_dir_;
    std::string barrier_dir_;
    std::size_t barrier_count_{0};

    /**
     * @brief Get path for a key-value file.
     *
     * @param key Key name
     */
    [[nodiscard]] std::string get_kv_path(std::string const& key) const;

    /**
     * @brief Get path for a barrier file.
     *
     * @param barrier_id Unique barrier identifier.
     */
    [[nodiscard]] std::string get_barrier_path(std::size_t barrier_id) const;

    /**
     * @brief Get path for rank alive file.
     *
     * @param rank Rank to retrieve file.
     */
    [[nodiscard]] std::string get_rank_alive_path(Rank rank) const;

    /**
     * @brief Wait for a file to exist.
     *
     * @param path File path.
     * @param timeout Timeout duration.
     * @return true if file exists within timeout, false otherwise.
     */
    bool wait_for_file(
        std::string const& path, Duration timeout = std::chrono::seconds{30}
    );

    /**
     * @brief Write string to file atomically.
     *
     * @param path Path to file.
     * @param content Content to write.
     */
    void write_file(std::string const& path, std::string const& content);

    /**
     * @brief Read string from file.
     *
     * @param path Path to file.
     */
    std::string read_file(std::string const& path);

    /**
     * @brief Clean up coordination directory after all ranks are done.
     *
     * This method can be called by all ranks, but only rank 0 performs the actual
     * cleanup. Rank 0 waits for all other ranks to finish before removing the
     * coordination directory. Non-zero ranks will return immediately (no-op).
     */
    void cleanup_coordination_directory();
};

}  // namespace rapidsmpf::bootstrap::detail
