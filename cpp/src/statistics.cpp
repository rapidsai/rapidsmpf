/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iomanip>
#include <sstream>

#include <rapidsmpf/statistics.hpp>

namespace rapidsmpf {

void Statistics::FormatterDefault(std::ostream& os, std::size_t count, double val) {
    os << val;
    if (count > 1) {
        os << " (avg " << (val / count) << ")";
    }
};

Statistics::Stat Statistics::get_stat(std::string const& name) const {
    RAPIDSMPF_LOCK_GUARD(mutex_);
    return stats_.at(name);
}

double Statistics::add_stat(
    std::string const& name, double value, Formatter const& formatter
) {
    if (!enabled()) {
        return 0;
    }
    RAPIDSMPF_LOCK_GUARD(mutex_);
    auto it = stats_.find(name);
    if (it == stats_.end()) {
        it = stats_.insert({name, Stat(formatter)}).first;
    }
    return it->second.add(value);
}

std::size_t Statistics::add_bytes_stat(std::string const& name, std::size_t nbytes) {
    return add_stat(name, nbytes, [](std::ostream& os, std::size_t count, double val) {
        os << format_nbytes(val);
        if (count > 1) {
            os << " (avg " << format_nbytes(val / count) << ")";
        }
    });
}

Duration Statistics::add_duration_stat(std::string const& name, Duration seconds) {
    return Duration(add_stat(
        name,
        seconds.count(),
        [](std::ostream& os, std::size_t count, double val) {
            os << format_duration(val);
            if (count > 1) {
                os << " (avg " << format_duration(val / count) << ")";
            }
        }
    ));
}

std::string Statistics::report(std::string const& header) const {
    std::stringstream ss;
    ss << header;
    if (!enabled()) {
        ss << " disabled.";
        return ss.str();
    }
    RAPIDSMPF_LOCK_GUARD(mutex_);

    std::size_t max_length{0};
    for (auto const& [name, _] : stats_) {
        max_length = std::max(max_length, name.size());
    }
    ss << "\n";
    for (auto const& [name, stat] : stats_) {
        ss << " - " << std::setw(max_length + 3) << std::left << name + ": ";
        stat.formatter()(ss, stat.count(), stat.value());
        ss << "\n";
    }
    return ss.str();
}


}  // namespace rapidsmpf
