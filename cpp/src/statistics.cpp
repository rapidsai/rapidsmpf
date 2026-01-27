/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iomanip>
#include <ranges>
#include <sstream>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

// Setting `mr_ = nullptr` disables memory profiling.
Statistics::Statistics(bool enabled) : enabled_{enabled}, mr_{nullptr} {}

Statistics::Statistics(RmmResourceAdaptor* mr) : enabled_{true}, mr_{mr} {
    RAPIDSMPF_EXPECTS(
        mr != nullptr,
        "when enabling memory profiling, `mr` cannot be nullptr",
        std::invalid_argument
    );
}

std::shared_ptr<Statistics> Statistics::disabled() {
    static std::shared_ptr<Statistics> ret = std::make_shared<Statistics>(false);
    return ret;
}

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
        name, seconds.count(), [](std::ostream& os, std::size_t count, double val) {
            os << format_duration(val);
            if (count > 1) {
                os << " (avg " << format_duration(val / count) << ")";
            }
        }
    ));
}

std::vector<std::string> Statistics::list_stat_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ks = std::views::keys(stats_);
    return std::vector<std::string>{ks.begin(), ks.end()};
}

void Statistics::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.clear();
}

bool Statistics::is_memory_profiling_enabled() const {
    return mr_ != nullptr;
}

Statistics::MemoryRecorder::MemoryRecorder(
    Statistics* stats, RmmResourceAdaptor* mr, std::string name
)
    : stats_{stats}, mr_{mr}, name_{std::move(name)} {
    RAPIDSMPF_EXPECTS(stats_ != nullptr, "the statistics cannot be null");
    RAPIDSMPF_EXPECTS(mr != nullptr, "the memory resource cannot be null");
    mr_->begin_scoped_memory_record();
    main_record_ = mr_->get_main_record();
}

Statistics::MemoryRecorder::~MemoryRecorder() {
    if (stats_ == nullptr) {
        return;
    }
    auto const scope = mr_->end_scoped_memory_record();

    std::lock_guard<std::mutex> lock(stats_->mutex_);
    auto& record = stats_->memory_records_[name_];
    ++record.num_calls;
    record.scoped.add_scope(scope);
    record.global_peak = std::max(record.global_peak, scope.peak());
}

Statistics::MemoryRecorder Statistics::create_memory_recorder(std::string name) {
    if (mr_ == nullptr) {
        return MemoryRecorder{};
    }
    return MemoryRecorder{this, mr_, std::move(name)};
}

std::unordered_map<std::string, Statistics::MemoryRecord> const&
Statistics::get_memory_records() const {
    return memory_records_;
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
    ss << "\n";

    // Print memory profiling.
    ss << "Memory Profiling\n";
    ss << "----------------\n";
    if (mr_ == nullptr) {
        ss << "Disabled";
        return ss.str();
    }

    // Insert the memory records in a vector we can sort.
    std::vector<std::pair<std::string, MemoryRecord>> sorted_records{
        memory_records_.begin(), memory_records_.end()
    };

    // Insert the "main" record, which is the overall statistics from `mr_`.
    auto const main_record = mr_->get_main_record();
    sorted_records.emplace_back(
        "main (all allocations using RmmResourceAdaptor)",
        MemoryRecord{
            .scoped = main_record, .global_peak = main_record.peak(), .num_calls = 1
        }
    );

    // Sort based on peak memory.
    std::ranges::sort(sorted_records, [](auto const& a, auto const& b) {
        return a.second.scoped.peak() > b.second.scoped.peak();
    });
    ss << "Legends:\n"
       << "  ncalls - number of times the scope was executed.\n"
       << "  peak   - peak memory usage by the scope.\n"
       << "  g-peak - global peak memory usage during the scope's execution.\n"
       << "  accum  - total accumulated memory allocations by the scope.\n";
    ss << "\nOrdered by: peak (descending)\n\n";

    ss << std::right << std::setw(8) << "ncalls" << std::setw(12) << "peak"
       << std::setw(12) << "g-peak" << std::setw(12) << "accum"
       << "  filename:lineno(name)\n";

    // Print the sorted records.
    for (auto const& [name, record] : sorted_records) {
        ss << std::right << std::setw(8) << record.num_calls << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.peak()) << std::setw(12)
           << rapidsmpf::format_nbytes(record.global_peak) << std::setw(12)
           << rapidsmpf::format_nbytes(record.scoped.total()) << "  " << name << "\n";
    }
    ss << "\nLimitation:\n"
       << "  - A scope only tracks allocations made by the thread that entered it.\n";
    return ss.str();
}


}  // namespace rapidsmpf
