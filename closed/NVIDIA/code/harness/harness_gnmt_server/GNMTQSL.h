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
#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "system_under_test.h"
#include "test_settings.h"
#include "query_sample_library.h"

#include "glog/logging.h"

#include "common.h"

class SampleLibrary : public mlperf::QuerySampleLibrary
{
public:
    SampleLibrary(std::string name, std::string fileName);

    const std::string &Name() const { return mName; }
    size_t TotalSampleCount() { return m_SampleCount; }
    size_t PerformanceSampleCount() { return m_PerfSampleCount; }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples);
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples);

    void ResetAccuracyMetric() {}
    void UpdateAccuracyMetric(mlperf::QuerySampleIndex sample_index, void* response_data, size_t response_size) {};
    double GetAccuracyMetric() { return 0; }

    //!
    //! \brief Return data by data id
    //!
    std::string GetDataById(int id) 
    {
        CHECK(id < enSentence.size()); 
        return enSentence[id];
    }

    std::string HumanReadableAccuracyMetric(double metric_value) { return std::to_string(metric_value * 100.0) + "%"; }
private:
    const std::string mName;
    const std::string m_FileName;

    std::vector<std::string> enSentence;    //! Vector that contains entire corpus text

    size_t m_SampleCount{0};
    size_t m_PerfSampleCount{0};
};
