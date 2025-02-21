/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cstddef>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/communicator/communicator.hpp>

namespace rapidsmp {


/**
 * @brief Track statistics across rapidsmp operations.
 */
class Statistics {
  public:
    /**
     * @brief Stores statistics for a single peer.
     */
    struct PeerStats {
        std::size_t comm_count{0};  ///< Number of messages communicated.
        std::size_t comm_nbytes{0};  ///< Number of bytes communicated.

        /**
         * @brief Compares two PeerStats objects for equality.
         * @param o The other PeerStats object to compare with.
         * @return Answer.
         */
        bool operator==(PeerStats const& o) const noexcept {
            return comm_count == o.comm_count && comm_nbytes == o.comm_nbytes;
        }
    };

    /// @brief A default statistics object, which is disabled (all operations are no-ops).
    Statistics() = default;

    /**
     * @brief Constructs a new statistics object (enabled).
     *
     * @param comm The communicator to use.
     */
    Statistics(std::shared_ptr<Communicator> comm) : comm_{std::move(comm)} {
        RAPIDSMP_EXPECTS(comm_ != nullptr, "the communicator pointer cannot be NULL");
        peer_stats_.resize(comm_->nranks());
    }

    ~Statistics() noexcept = default;

    Statistics(const Statistics&) = delete;
    Statistics& operator=(const Statistics&) = delete;

    /**
     * @brief Move constructor.
     * @param o The Statistics object to move from.
     */
    Statistics(Statistics&& o) noexcept
        : comm_(o.comm_), peer_stats_{std::move(o.peer_stats_)} {}

    /**
     * @brief Move assignment operator.
     *
     * @param o The Statistics object to move from.
     * @return A reference to the updated Statistics object.
     */
    Statistics& operator=(Statistics&& o) noexcept {
        comm_ = o.comm_;
        peer_stats_ = std::move(o.peer_stats_);
        return *this;
    }

    /**
     * @brief Checks if statistics is enabled.
     *
     * Operations on disabled statistics is no-ops.
     *
     * @return True if the object is enabled, otherwise false.
     */
    bool enabled() const noexcept {
        return comm_ != nullptr;
    }

    /**
     * @brief Retrieves the statistics for a given peer.
     *
     * @param peer The rank of the peer to retrieve statistics for.
     * @return A PeerStats object for the specified peer.
     */
    PeerStats get_peer_stats(Rank peer) const {
        if (!enabled()) {
            return PeerStats{};
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return peer_stats_.at(peer);
    }

    /**
     * @brief Add peer communication to the statistics.
     *
     * This is a no-op if the statistics is disabled.
     *
     * @param peer The rank of the peer.
     * @param nbytes The number of bytes communicated.
     * @return The total number of bytes communicated to the peer after the update.
     */
    std::size_t add_peer_comm(Rank peer, std::size_t nbytes) {
        if (!enabled()) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto& p = peer_stats_.at(peer);
        ++p.comm_count;
        return p.comm_nbytes += nbytes;
    }

    /**
     * @brief Generates a report of statistics in a formatted string.
     *
     * @param column_width The width of each column in the report.
     * @param label_width The width of the labels in the report.
     * @return A string representing the formatted statistics report.
     */
    std::string report(int column_width = 12, int label_width = 22) const {
        if (!enabled()) {
            return "Statistics: disabled";
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::stringstream ss;
        ss << "Statistics:\n";
        ss << std::setw(label_width - 3) << std::left << " - peers:";
        for (Rank i = 0; i < comm_->nranks(); ++i) {
            ss << std::right << std::setw(column_width) << "Rank" << i;
        }
        ss << "\n" << std::setw(label_width) << std::left << " - comm-gpu-data-total:";
        for (Rank i = 0; i < comm_->nranks(); ++i) {
            ss << std::right << std::setw(column_width)
               << format_nbytes(peer_stats_.at(i).comm_nbytes) << " ";
        }
        ss << "\n" << std::setw(label_width) << std::left << " - comm-gpu-data-mean:";
        for (Rank i = 0; i < comm_->nranks(); ++i) {
            ss << std::right << std::setw(column_width)
               << format_nbytes(
                      peer_stats_.at(i).comm_nbytes / (double)peer_stats_.at(i).comm_count
                  )
               << " ";
        }
        ss << "\n";
        return ss.str();
    }

  private:
    mutable std::mutex mutex_;
    std::shared_ptr<Communicator> comm_;
    std::vector<PeerStats> peer_stats_;
};

}  // namespace rapidsmp
