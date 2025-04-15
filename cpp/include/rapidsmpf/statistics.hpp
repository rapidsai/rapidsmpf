/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>

#include <rapidsmpf/buffer/buffer.hpp>
#include <rapidsmpf/communicator/communicator.hpp>

namespace rapidsmpf {


/**
 * @class Statistics
 * @brief Track statistics across rapidsmpf operations.
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
     * @param header The header to use for the report.
     * @return Formatted statistics report as a string.
     */
    std::string report(std::string const& header = "Statistics:") const;

    /**
     * @brief Function type for formatting statistics output.
     *
     * Must take an ostream, counter, and value.
     */
    using Formatter = std::function<void(std::ostream&, std::size_t, double)>;

    /**
     * @brief Default formatter for statistics output (implements `Formatter`).
     *
     * Outputs the value as-is and an average if the statistic consist of the sum
     * of multiple values.
     *
     * @param os The output stream to write to.
     * @param count The number of elements contributing to the value.
     * @param val The total value to be formatted.
     */
    static void FormatterDefault(std::ostream& os, std::size_t count, double val);

    /**
     * @brief Represents an individual statistic entry.
     */
    class Stat {
      public:
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
            return count_ == o.count() && value_ == o.value();
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

        /**
         * @brief Returns the number of times the statistic was updated.
         *
         * @return The update count.
         */
        [[nodiscard]] std::size_t count() const noexcept {
            return count_;
        }

        /**
         * @brief Returns the current accumulated value of the statistic.
         *
         * @return The accumulated value.
         */
        [[nodiscard]] double value() const noexcept {
            return value_;
        }

        /**
         * @brief Returns the formatter function associated with this statistic.
         *
         * @return Reference to the formatter function.
         */
        [[nodiscard]] Formatter const& formatter() const noexcept {
            return formatter_;
        }

      private:
        std::size_t count_{0};
        double value_{0};
        Formatter formatter_;
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
        Formatter const& formatter = FormatterDefault
    );

    /**
     * @brief Adds a byte count to the statistics.
     *
     * Convenience function that calls `add_stat` with a formatter suitable for byte
     * counters.
     *
     * @param name Name of the statistic.
     * @param nbytes Number of bytes.
     * @return Updated total number of bytes.
     */
    std::size_t add_bytes_stat(std::string const& name, std::size_t nbytes);

    /**
     * @brief Adds a duration to the statistics.
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
    mutable std::mutex mutex_;
    bool enabled_;
    std::map<std::string, Stat> stats_;
};
}  // namespace rapidsmpf
