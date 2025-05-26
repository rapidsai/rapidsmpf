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
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.at(name);
}

double Statistics::add_stat(
    std::string const& name, double value, Formatter const& formatter
) {
    if (!enabled()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
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

Statistics::MemoryRecorder::MemoryRecorder(
    Statistics* stats, rmm_statistics_resource* mr, std::string name
)
    : stats_{stats}, mr_{mr}, name_{std::move(name)} {
    RAPIDSMPF_EXPECTS(stats_ != nullptr, "the statistics cannot be null");
    RAPIDSMPF_EXPECTS(mr != nullptr, "the memory resource cannot be null");
    mr_->push_counters();
}

Statistics::MemoryRecorder::~MemoryRecorder() {
    std::lock_guard<std::mutex> lock(stats_->mutex_);
    auto bytes_counter = mr_->pop_counters().first;
    auto& record = stats_->memory_records_[name_];
    ++record.num_calls;
    record.total += bytes_counter.total;
    record.peak = std::max(record.peak, bytes_counter.peak);
}

std::string Statistics::report(std::string const& header) const {
    std::stringstream ss;
    ss << header;
    if (!enabled()) {
        ss << " disabled.";
        return ss.str();
    }
    std::lock_guard<std::mutex> lock(mutex_);

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

    // Print memory profiling.
    ss << "Memory Profiling\n";
    ss << "================\n";

    if (memory_records_.empty()) {
        ss << "No data, maybe memory profiling wasn't enabled?";
        return ss.str();
    }

    ss << "Legends:\n"
       << "  ncalls       - number of times the function or code block was called.\n"
       << "  peak  - peak memory allocated in function or code block (in bytes).\n"
       << "  total - total memory allocated in function or code block (in "
          "bytes).\n";

    ss << "\nOrdered by: "
       << "TODO"
       << "\n\n";
    ss << "ncalls     peak    total  filename:lineno(function)\n";
    for (auto const& [name, record] : memory_records_) {
        ss << std::right << std::setw(6) << record.num_calls << " " << std::right
           << std::setw(15) << record.peak << " " << std::right << std::setw(15)
           << record.total << "  " << name << "\n";
    }
    return ss.str();
}


}  // namespace rapidsmpf
