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
#include "GNMTQSL.h"

SampleLibrary::SampleLibrary(std::string name, std::string fileName) : mName{name}, m_FileName{fileName}
{
    std::ifstream inputEnFile(m_FileName);
    if (!inputEnFile)
    {
        LOG(INFO) << "Error opening input file " << m_FileName;
        CHECK(false);
    }
    std::string line;
    while (std::getline(inputEnFile, line))
    {
        if (!line.empty())
        {
            ++m_SampleCount;
            ++m_PerfSampleCount;
        }
    }
}

void SampleLibrary::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples)
{
    std::string line;
    std::ifstream inputEnFile(m_FileName);
    std::vector<std::string> allSentences;
    if (!inputEnFile)
    {
        LOG(INFO) << "Error opening input file " << m_FileName;
        CHECK(false);
    }

    while (std::getline(inputEnFile, line))
    {
        enSentence.push_back(line);
    }

}

void SampleLibrary::UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples)
{
    // No op at this point
}
