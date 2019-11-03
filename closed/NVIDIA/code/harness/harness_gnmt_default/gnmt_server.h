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

#include "query_sample_library.h"
#include "system_under_test.h"

#include "GNMTCore.h"

#include <map>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>

class GNMTServer : public mlperf::QuerySampleLibrary, public mlperf::SystemUnderTest
{
public:
    GNMTServer(
        std::string name,
        std::string fileName,
        std::shared_ptr<Config> config,
        std::string engineDir,
        const std::vector<int>& gpus);

    virtual ~GNMTServer();

    const std::string& Name() const override;

    size_t TotalSampleCount() override;
    size_t PerformanceSampleCount() override;
    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;
    void ReportLatencyResults( const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override;

private:
    // If the function returns empty vector then there are no tasks remained and the caller should exit
    std::vector<mlperf::QuerySample> GetTasks(int maxSampleCount);

    void ProcessTasks(std::shared_ptr<GNMTCore>, int deviceId);

private:
    const std::string mName;
    const std::string mFileName;
    int mMaxBatchSize;

    size_t mSampleCountInFile;
    std::map<mlperf::QuerySampleIndex, std::string> mInputSentences;

    std::deque<mlperf::QuerySample> mTasks;

	// mutex to serialize access to mTasks member variable
    std::mutex mMtx;

	// The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::condition_variable mCondVar;

	// Indicates that there will no new tasks and the worker threads should stop processing samples 
    bool mStopWork;

    std::vector<std::thread> mWorkerThreads;
};
