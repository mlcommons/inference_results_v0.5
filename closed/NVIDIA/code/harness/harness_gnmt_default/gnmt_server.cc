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

#include "gnmt_server.h"

#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <set>

GNMTServer::GNMTServer(
    std::string name,
    std::string fileName,
    std::shared_ptr<Config> config,
    std::string engineDir,
    const std::vector<int>& gpus)
    : mName{name}
    , mFileName{fileName}
    , mMaxBatchSize{config->maxBatchSize}
    , mSampleCountInFile{0}
    , mStopWork{false}
{
    std::ifstream inputEnFile(mFileName);
    if (!inputEnFile)
    {
        LOG(INFO) << "Error opening input file " << mFileName;
        CHECK(false);
    }
    std::string line;
    while (std::getline(inputEnFile, line))
    {
        if (!line.empty())
            ++mSampleCountInFile;
    }

    mWorkerThreads.reserve(gpus.size());
    for(auto deviceId: gpus)
    {
        CHECK_CUDA(cudaSetDevice(deviceId));
        auto gnmtCore = std::make_shared<GNMTCore>(config, true, false, engineDir, false);
        gnmtCore->setup();
        mWorkerThreads.emplace_back(&GNMTServer::ProcessTasks, this, gnmtCore, deviceId);
    }
}

GNMTServer::~GNMTServer()
{
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }
    for(auto& workerThread: mWorkerThreads)
        workerThread.join();
}

const std::string& GNMTServer::Name() const
{
    return mName;
}

size_t GNMTServer::TotalSampleCount()
{
    return mSampleCountInFile;
}

size_t GNMTServer::PerformanceSampleCount()
{
    return mSampleCountInFile;
}

void GNMTServer::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples)
{
    std::ifstream inputEnFile(mFileName);
    if (!inputEnFile)
    {
        LOG(INFO) << "Error opening input file " << mFileName;
        CHECK(false);
    }
    std::string line;
    mlperf::QuerySampleIndex currentLineId = 0;
    // Creating set collection of sample IDs to load,
    // in order to quickly lookup if we need to load a sample when iterating over input file
    std::set<mlperf::QuerySampleIndex> samplesToLoad(samples.begin(), samples.end());
    while (std::getline(inputEnFile, line))
    {
        if (!line.empty())
        {
            // We are loading strings as is, without any processing
            if (samplesToLoad.find(currentLineId) != samplesToLoad.end())
                mInputSentences.insert(std::make_pair(currentLineId, line));
            ++currentLineId;
        }
    }
}

void GNMTServer::UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples)
{
    for(auto elem: samples)
    {
        mInputSentences.erase(elem);
    }
}

void GNMTServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    // Sort samples in the descending order of sentence length
    {
        std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
        for (int samplePos = 0; samplePos < samples.size(); ++samplePos)
            sequenceSamplePosAndLength[samplePos] = std::make_pair(samplePos, static_cast<int>(mInputSentences[samples[samplePos].index].length()));
        std::sort(
            sequenceSamplePosAndLength.begin(),
            sequenceSamplePosAndLength.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });

        for (int beginSamplePos = 0; beginSamplePos < sequenceSamplePosAndLength.size(); beginSamplePos += mMaxBatchSize)
        {
            int actualBatchSize = std::min(mMaxBatchSize, static_cast<int>(sequenceSamplePosAndLength.size()) - beginSamplePos);
            {
                std::unique_lock<std::mutex> lck(mMtx);
                for(int i = 0; i < actualBatchSize; ++i)
                {
                    int samplePosInOriginalRequest = sequenceSamplePosAndLength[beginSamplePos + i].first;
                    mTasks.push_back(samples[samplePosInOriginalRequest]);
                }

                // Let some worker thread to consume tasks
                mCondVar.notify_one();
            }
        }
    }
}

void GNMTServer::FlushQueries()
{
    // Nothing to do for the function
}

void GNMTServer::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns)
{
    // Nothing to do for the function
}

void GNMTServer::ProcessTasks(std::shared_ptr<GNMTCore> gnmtCore, int deviceId)
{
    CHECK_CUDA(cudaSetDevice(deviceId));

    auto tasks = GetTasks(mMaxBatchSize);
    // Process samples in batches
    while (!tasks.empty())
    {
        int actualBatchSize = tasks.size();
        std::vector<std::string> currentSamples(actualBatchSize);
        for(int i = 0; i < actualBatchSize; ++i)
            currentSamples[i] = mInputSentences[tasks[i].index];

        auto tokenWords = gnmtCore->translate(currentSamples, true);

        std::vector<mlperf::QuerySampleResponse> responses(actualBatchSize);
        std::vector<std::string> responseStrings(actualBatchSize);
        for(int i = 0; i < actualBatchSize; ++i)
        {
            std::stringstream translatedText;
            writeTokenizedSentence(translatedText, tokenWords[i]);
            responseStrings[i] = translatedText.str();
            uintptr_t data = reinterpret_cast<mlperf::ResponseId>(responseStrings[i].c_str());
            responses[i] = mlperf::QuerySampleResponse{tasks[i].id, data, responseStrings[i].length()};
        }

        mlperf::QuerySamplesComplete(&responses.front(), actualBatchSize);
        tasks = GetTasks(mMaxBatchSize);
    }
}

std::vector<mlperf::QuerySample> GNMTServer::GetTasks(int maxSampleCount)
{
    std::vector<mlperf::QuerySample> res;
    res.reserve(maxSampleCount);
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck(mMtx);
    mCondVar.wait(lck, [&] {return (!mTasks.empty()) || mStopWork;} );

    // Consume up to maxSampleCount tasks
    for(int i = 0; (i < maxSampleCount) && !mTasks.empty(); ++i)
    {
        res.push_back(mTasks.front());
        mTasks.pop_front();
    }

    // Let some other thread to consume more tasks if this one got any
    if (!res.empty())
        mCondVar.notify_one();

    return res;
}
