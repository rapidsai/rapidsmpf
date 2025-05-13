/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::config {

class Option {
  public:
    virtual ~Option() = default;
};

namespace detail {
class OptionsImpl {
  public:
    OptionsImpl(
        std::unordered_map<std::string, std::unique_ptr<Option>> options,
        std::unordered_map<std::string, std::string> options_as_strings
    );

    template <typename T>
    T const* get(std::string const& key) {
        static_assert(std::is_base_of<Option, T>::value, "T must derive from Option");

        if (options_.find(key) == options_.end()) {
            if (options_as_strings_.find(key) == options_as_strings_.end()) {
                options_[key] = std::make_unique<T>();
            } else {
                options_[key] = std::make_unique<T>(options_as_strings_[key]);
            }
        }

        auto option = dynamic_cast<T*>(options_.at(key).get());
        RAPIDSMPF_EXPECTS(
            option != nullptr,
            "accessing option with incompatible template type",
            std::invalid_argument
        );
        return option;
    }

  private:
    std::unordered_map<std::string, std::unique_ptr<Option>> options_;
    std::unordered_map<std::string, std::string> options_as_strings_;
};
}  // namespace detail

class Options {
  public:
    Options(
        std::unordered_map<std::string, std::unique_ptr<Option>> options_ = {},
        std::unordered_map<std::string, std::string> options_as_strings_ = {}
    );

    template <typename T>
    T const* get(std::string const& key) {
        return impl_->get<T>(key);
    }

  private:
    std::shared_ptr<detail::OptionsImpl> impl_;
};


}  // namespace rapidsmpf::config
