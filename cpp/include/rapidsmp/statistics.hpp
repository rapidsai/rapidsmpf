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
#include <mutex>
#include <string>

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
        std::size_t payload_send_count{0};  ///< Number of payload messages send.
        std::size_t payload_send_nbytes{0};  ///< Number of bytes of payload send.
        std::size_t payload_recv_count{0};  ///< Number of payload messages received.
        std::size_t payload_recv_nbytes{0};  ///< Number of bytes of payload received.

        /**
         * @brief Compares for equality.
         * @param o The other object to compare with.
         * @return Answer.
         */
        bool operator==(PeerStats const& o) const noexcept {
            return payload_send_count == o.payload_send_count
                   && payload_send_nbytes == o.payload_send_nbytes
                   && payload_recv_count == o.payload_recv_count
                   && payload_recv_nbytes == o.payload_recv_nbytes;
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
        comm_ = std::move(o.comm_);
        peer_stats_ = std::move(o.peer_stats_);
        return *this;
    }

    /**
     * @brief Checks if statistics is enabled.
     *
     * All operations are no-ops when statistics is disabled.
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
    PeerStats get_peer_stats(Rank peer) const;

    /**
     * @brief Add payload sent to specified peer.
     *
     * This is a no-op if the statistics is disabled.
     *
     * @param peer The rank of the peer.
     * @param nbytes The number of bytes sent.
     * @return The total payload sent to the peer after the update.
     */
    std::size_t add_payload_send(Rank peer, std::size_t nbytes);

    /**
     * @brief Add payload received from specified peer.
     *
     * This is a no-op if the statistics is disabled.
     *
     * @param peer The rank of the peer.
     * @param nbytes The number of bytes received.
     * @return The total payload received from the peer after the update.
     */
    std::size_t add_payload_recv(Rank peer, std::size_t nbytes);

    /**
     * @brief Generates a report of statistics in a formatted string.
     *
     * @param column_width The width of each column in the report.
     * @param label_width The width of the labels in the report.
     * @return A string representing the formatted statistics report.
     */
    std::string report(int column_width = 12, int label_width = 30) const;

  private:
    mutable std::mutex mutex_;
    std::shared_ptr<Communicator> comm_;
    std::vector<PeerStats> peer_stats_;
};

}  // namespace rapidsmp
