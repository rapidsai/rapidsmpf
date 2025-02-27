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
#include <utility>

#include <rapidsmp/buffer/buffer.hpp>
#include <rapidsmp/communicator/communicator.hpp>

namespace rapidsmp {


/**
 * @class Statistics
 * @brief Track statistics across rapidsmp operations.
 */
class Statistics {
  public:
    /**
     * @brief Constructs a new Statistics object.
     *
     * @param enabled Whether statistics tracking is enabled.
     */
    Statistics(bool enabled = true) : enabled_{enabled} {}

    ~Statistics() noexcept = default;
    Statistics(const Statistics&) = delete;
    Statistics& operator=(const Statistics&) = delete;

    /**
     * @brief Move constructor.
     *
     * @param o The Statistics object to move from.
     */
    Statistics(Statistics&& o) noexcept
        : enabled_(o.enabled_), stats_{std::move(o.stats_)} {}

    /**
     * @brief Move assignment operator.
     *
     * @param o The Statistics object to move from.
     * @return Reference to the updated Statistics object.
     */
    Statistics& operator=(Statistics&& o) noexcept {
        enabled_ = o.enabled_;
        stats_ = std::move(o.stats_);
        return *this;
    }

    /**
     * @brief Checks if statistics tracking is enabled.
     *
     * @return True if enabled, otherwise false.
     */
    bool enabled() const noexcept {
        return enabled_;
    }

    /**
     * @brief Generates a formatted report of collected statistics.
     *
     * @param column_width Width of each column.
     * @param label_width Width of the labels.
     * @return Formatted statistics report as a string.
     */
    std::string report(int column_width = 12, int label_width = 30) const;

    /**
     * @brief Function type for formatting statistics output.
     *
     * Must take an ostream, counter, and value.
     */
    using Formatter = std::function<void(std::ostream&, std::size_t, double)>;

    /**
     * @brief Represents an individual statistic entry.
     */
    struct Stat {
        /**
         * @brief Constructs a statistic with specified formatter.
         *
         * @param formatter The formatter function.
         */
        Stat(Formatter formatter) : formatter_{std::move(formatter)} {}

        /**
         * @brief Equality comparison operator.
         *
         * @param o Other statistic object.
         * @return True if equal, otherwise false.
         */
        bool operator==(Stat const& o) const noexcept {
            return count_ == o.count_ && value_ == o.value_;
        }

        /**
         * @brief Adds a value to the statistic.
         *
         * @param value Value to add.
         * @return Updated total value.
         */
        double add(double value) {
            ++count_;
            return value_ += value;
        }

        std::size_t count_{0};  ///< Number of times the statistic was updated.
        double value_{0};  ///< Accumulated value.
        Formatter formatter_;  ///< Formatter function.
    };

    /**
     * @brief Retrieves a statistic by name.
     *
     * @param name Name of the statistic.
     * @return The requested statistic.
     */
    Stat get_stat(std::string const& name) const;

    /**
     * @brief Adds a value to a statistic.
     *
     * @param name Name of the statistic.
     * @param value Value to add.
     * @param formatter Formatter function, which by default formats the value as-is.
     * @return Updated total value.
     */
    double add_stat(
        std::string const& name,
        double value,
        Formatter const& formatter =
            [](std::ostream& os, std::size_t count, double val) {
                os << val;
                if (count > 1) {
                    os << " (avg " << (val / count) << ")";
                }
            }
    );

    /**
     * @brief Adds a byte value to a statistic.
     *
     * Convenience function that calls `add_stat` with a formatter suitable for byte
     * counters.
     *
     * @param name Name of the statistic.
     * @param nbytes Number of bytes.
     * @return Updated total value.
     */
    std::size_t add_bytes_stat(std::string const& name, std::size_t nbytes);

    /**
     * @brief Adds a byte value to a statistic.
     *
     * Convenience function that calls `add_stat` with a formatter suitable for
     * durations.
     *
     * @param name Name of the statistic.
     * @param seconds The duration in seconds.
     * @return Updated total value.
     */
    Duration add_duration_stat(std::string const& name, Duration seconds);


  private:
    mutable std::mutex mutex_;  ///< Mutex for thread safety.
    bool enabled_;  ///< Whether statistics tracking is enabled.
    std::map<std::string, Stat> stats_;  ///< Map of statistics by name.
};
}  // namespace rapidsmp
