/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <unordered_set>

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

std::shared_ptr<Statistics> Statistics::from_options(
    RmmResourceAdaptor* mr, config::Options options
) {
    bool const statistics = options.get<bool>("statistics", [](auto const& s) {
        return parse_string<bool>(s.empty() ? "False" : s);
    });
    return statistics ? std::make_shared<Statistics>(mr) : Statistics::disabled();
}

std::shared_ptr<Statistics> Statistics::disabled() {
    static std::shared_ptr<Statistics> ret = std::make_shared<Statistics>(false);
    return ret;
}

Statistics::Stat Statistics::get_stat(std::string const& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.at(name);
}

void Statistics::add_stat(std::string const& name, double value) {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, _] = stats_.try_emplace(name);
    it->second.add(value);
}

bool Statistics::exist_report_entry_name(std::string const& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return formatters_.contains(name);
}

void Statistics::register_formatter(std::string const& name, Formatter formatter) {
    if (!enabled() || exist_report_entry_name(name)) {
        return;
    }
    register_formatter(name, {name}, std::move(formatter));
}

void Statistics::register_formatter(
    std::string const& report_entry_name,
    std::vector<std::string> const& stat_names,
    Formatter formatter
) {
    if (!enabled() || exist_report_entry_name(report_entry_name)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    formatters_.try_emplace(report_entry_name, stat_names, std::move(formatter));
}

void Statistics::add_bytes_stat(std::string const& name, std::size_t nbytes) {
    if (!exist_report_entry_name(name)) {
        register_formatter(name, [](std::ostream& os, std::vector<Stat> const& stats) {
            auto const val = stats[0].value();
            auto const count = stats[0].count();
            os << format_nbytes(val);
            if (count > 1) {
                os << " | avg " << format_nbytes(val / count);
            }
        });
    }
    add_stat(name, static_cast<double>(nbytes));
}

void Statistics::add_duration_stat(std::string const& name, Duration seconds) {
    if (!exist_report_entry_name(name)) {
        register_formatter(name, [](std::ostream& os, std::vector<Stat> const& stats) {
            auto const val = stats[0].value();
            auto const count = stats[0].count();
            os << format_duration(val);
            if (count > 1) {
                os << " | avg " << format_duration(val / count);
            }
        });
    }
    add_stat(name, seconds.count());
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

    // Reporting strategy:
    //
    // Each registered formatter claims one or more stat names and renders them into a
    // single labelled entry line using a custom function.  Any stat that is not claimed
    // by any formatter is rendered with a plain numeric default entry line.  All entry
    // lines are then sorted alphabetically by their label and printed together.

    using EntryLine = std::pair<std::string, std::string>;

    std::vector<EntryLine> lines;
    std::unordered_set<std::string> consumed;

    // Returns true only if every stat name required by a formatter has been recorded.
    // If false, the entry is rendered as "No data collected".
    auto has_all_stats = [&](auto const& names) {
        return std::ranges::all_of(names, [&](auto const& sname) {
            return stats_.contains(sname);
        });
    };

    // Formatter-based lines. Emit "No data collected" if any required stats are missing.
    for (auto const& [report_entry_name, entry] : formatters_) {
        if (!has_all_stats(entry.stat_names)) {
            lines.emplace_back(report_entry_name, "No data collected");
            continue;
        }

        std::vector<Stat> stat_vec;
        stat_vec.reserve(entry.stat_names.size());
        for (auto const& sname : entry.stat_names) {
            stat_vec.push_back(stats_.at(sname));
        }

        for (auto const& sname : entry.stat_names) {
            consumed.insert(sname);
        }

        std::ostringstream line;
        entry.fn(line, stat_vec);
        lines.emplace_back(report_entry_name, std::move(line).str());
    }

    // Uncovered stats get a default raw-value format.
    for (auto const& [name, stat] : stats_) {
        if (consumed.contains(name)) {
            continue;
        }
        std::ostringstream line;
        line << stat.value();
        if (stat.count() > 1) {
            line << " (count " << stat.count() << ")";
        }
        lines.emplace_back(name, std::move(line).str());
    }

    std::ranges::sort(lines, {}, &EntryLine::first);
    std::size_t max_length = 0;
    for (auto const& [name, _] : lines) {
        max_length = std::max(max_length, name.size());
    }

    ss << "\n";
    for (auto const& [name, text] : lines) {
        ss << " - " << std::setw(max_length + 3) << std::left << name + ": " << text
           << "\n";
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

void Statistics::record_copy(MemoryType src, MemoryType dst, std::size_t nbytes) {
    using Key = std::pair<MemoryType, MemoryType>;
    // Use a lambda to construct all stat names once at first call.
    static std::map<Key, std::string> const name_map = [] {
        std::map<Key, std::string> ret;
        for (MemoryType s : MEMORY_TYPES) {
            auto const src_name = to_lower(to_string(s));
            for (MemoryType d : MEMORY_TYPES) {
                auto const dst_name = to_lower(to_string(d));
                ret.emplace(Key{s, d}, "copy-" + src_name + "-to-" + dst_name);
            }
        }
        return ret;
    }();
    add_bytes_stat(name_map.at({src, dst}), nbytes);
}

}  // namespace rapidsmpf
