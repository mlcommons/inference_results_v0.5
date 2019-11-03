// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <algorithm>
#include <thread>
#include <utility>
#include <vector>
#include <map>

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<std::string> parseDevices(const std::string& device_string) {
    std::string comma_separated_devices = device_string;
    if (comma_separated_devices.find(":") != std::string::npos) {
        comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
    }
    auto devices = split(comma_separated_devices, ',');
    for (auto& device : devices)
        device = device.substr(0, device.find("("));
    return devices;
}

std::map<std::string, uint32_t> parseValuePerDevice(const std::vector<std::string>& devices,
                                                    const std::string& values_string) {
    //  Format: <device1>:<value1>,<device2>:<value2> or just <value>
    auto values_string_upper = values_string;
    std::transform(values_string_upper.begin(),
                   values_string_upper.end(),
                   values_string_upper.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    std::map<std::string, uint32_t> result;
    auto device_value_strings = split(values_string_upper, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec =  split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto it = std::find(devices.begin(), devices.end(), device_value_vec.at(0));
            if (it != devices.end()) {
                result[device_value_vec.at(0)] = std::stoi(device_value_vec.at(1));
            }
        } else if (device_value_vec.size() == 1) {
            uint32_t value = std::stoi(device_value_vec.at(0));
            for (auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}
