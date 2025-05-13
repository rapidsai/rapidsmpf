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

/**
 * @brief Base class for configuration options.
 *
 * All configuration options must derive from this class.
 */
class Option {
  public:
    /**
     * @brief Virtual destructor for the `Option` base class.
     */
    virtual ~Option() = default;
};

namespace detail {

/**
 * @brief Internal implementation of the `Options` class.
 *
 * This class is used internally by `Options` to manage the storage and retrieval
 * of configuration options. Refer to the `Options` class documentation for details.
 */
class OptionsImpl {
  public:
    /**
     * @brief Constructs an `OptionsImpl` instance.
     *
     * @param options_as_strings A map of option keys to their string representations.
     * @param options A map of option keys to their corresponding `Option` objects.
     */
    OptionsImpl(
        std::unordered_map<std::string, std::string> options_as_strings,
        std::unordered_map<std::string, std::unique_ptr<Option>> options
    );

    /**
     * @brief Retrieves a configuration option by key.
     *
     * Refer to the `Options::get` method for usage details.
     *
     * @tparam T The type of the option to retrieve. Must derive from `Option`.
     * @param key The key of the option to retrieve.
     * @return A pointer to the retrieved option.
     * @throws std::invalid_argument If the option exists but is of an incompatible type.
     */
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

        auto option = dynamic_cast<T*>(options_[key].get());
        RAPIDSMPF_EXPECTS(
            option != nullptr,
            "accessing option with incompatible template type",
            std::invalid_argument
        );
        return option;
    }

  private:
    std::unordered_map<std::string, std::string> options_as_strings_;
    std::unordered_map<std::string, std::unique_ptr<Option>> options_;
};

}  // namespace detail

/**
 * @brief Manages configuration options for the RapidsMPF operations.
 *
 * The `Options` class provides a high-level interface for storing and retrieving
 * configuration options.
 *
 * To avoid having to use `std::shared_ptr<Options>` arguments everywhere, it uses an
 * internal implementation (`OptionsImpl`) to handle the actual storage and retrieval
 * logic.
 */
class Options {
  public:
    /**
     * @brief Constructs an `Options` instance.
     *
     * @param options_as_strings A map of option keys to their string representations.
     * @param options A map of option keys to their corresponding `Option` objects.
     */
    Options(
        std::unordered_map<std::string, std::string> options_as_strings = {},
        std::unordered_map<std::string, std::unique_ptr<Option>> options = {}
    );

    /**
     * @brief Retrieves a configuration option by key.
     *
     * If the option does not exist, it is created using:
     *  1) its string representation if available, or
     *  2) using its default constructor.
     *
     * @tparam T The type of the option to retrieve. Must derive from `Option`.
     * @param key The key of the option to retrieve.
     * @return A pointer to the retrieved option.
     * @throws std::invalid_argument If the option exists but is of an incompatible type.
     */
    template <typename T>
    T const* get(std::string const& key) {
        return impl_->get<T>(key);
    }

  private:
    std::shared_ptr<detail::OptionsImpl> impl_;
};

}  // namespace rapidsmpf::config
