/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


#ifndef __CALLBACK_HPP__
#define __CALLBACK_HPP__

#include <functional>
#include <map>
#include <iostream>

#include "query_sample_library.h"

void cocoCallback(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)
{
    for (size_t i = 0; i < response_count; i++)
    {
        auto &r = responses[i];
        size_t size = r.size;
        size_t maxKeepCount = (size / sizeof(float) - 1) / 7;
        int32_t keepCount = *(reinterpret_cast<int32_t*>(r.data) + maxKeepCount * 7);
        r.size = static_cast<size_t>(keepCount * sizeof(float) * 7);
        float image_id = sample_ids[i];
        for (size_t k = 0; k < keepCount; k++) {
          *(reinterpret_cast<float*>(r.data) + k * 7) = image_id;
        }
    }
}

/* Define a map for post-processing callback functions */
std::map<std::string, std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)>> callbackMap = {
    {"", nullptr},
    {"coco", cocoCallback}
};

#endif /* __CALLBACK_HPP__ */
