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

#include "test_settings.h"
#include "system_under_test.h"
#include "query_sample_library.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "moodycamel/blockingconcurrentqueue.h"
using moodycamel::BlockingConcurrentQueue;
using moodycamel::ConsumerToken;
using moodycamel::ProducerToken;

#include "glog/logging.h"

#include "tensorrt/laboratory/core/affinity.h"
#include "tensorrt/laboratory/core/thread_pool.h"
#include "tensorrt/laboratory/infer_bench.h"
#include "tensorrt/laboratory/infer_runner.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/core/pool.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "params.h"
#include "GNMTModel.h"
#include "GNMTInferRunner.h"
#include "GNMTBindings.h"
#include "GNMTBench.h"
#include "common.h"
#include "GNMTQSL.h"

using trtlab::Affinity;
using trtlab::ThreadPool;
using trtlab::TensorRT::Bindings;
using trtlab::TensorRT::InferBench;
using trtlab::TensorRT::InferenceManager;
using trtlab::TensorRT::InferRunner;
using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::StandardRuntime;
using trtlab::TensorRT::Model;

using trtlab::TensorRT::GNMTInferRunner;
using trtlab::Pool;

struct WorkPacket
{
    mlperf::QuerySample sample;
    std::chrono::high_resolution_clock::time_point received_time;
};

class SingleGPUServer
{
public:
    SingleGPUServer(int device, const std::chrono::microseconds latency_bound): mDevice{device}, mLatencyBound{latency_bound} {}

    //!
    //! \brief Set up GNMT benchmark search range
    //!
    void SetBenchMarkRange(int batch_min, int batch_inc)
    {
        mBenchmarkBatchMin = batch_min;
        mBenchmarkBatchInc = batch_inc;
    }

    //!
    //! \brief Set up GNMT benchmark input file
    //!
    void SetBenMarkInputFile(std::string input_file)
    {
        mBenchmarkInputFile = input_file;
    }

    //!
    //! \brief Deserilaze engine files and initialize SingleGPUServer
    //!
    void Setup(std::vector<std::string> engine_dirs, int concurrency);

    //!
    //! \brief Run benchmark to get runtime by batch size
    //!
    void BenchMark(std::map<double, uint32_t> &batch_size_by_elapsed_time, std::map<uint32_t, double> &estimated_execution_time_by_batch_size);
    
    //!
    //! \brief Running inference on this gpu server
    //! 
    //! \param work_packets Vectors of query samples
    //! \param qsl Pointer to SampleLibrary so that we can access real data by providing data id
    //! \param running_batch_size Batch size for execution
    //!
    void Infer(std::vector<WorkPacket> &work_packets, std::shared_ptr<SampleLibrary> qsl, int running_batch_size);

    //!
    //! \brief Initialize SingleGPUServer
    //!
    void Initialize(std::shared_ptr<InferenceManager> resources)
    {
        mResources = std::move(resources);
    }

    void Reset()
    {
        mResources.reset();
    }

    int getMaxBatchSize()
    {
        return mMaxBatchSize;
    }

    auto getExtraResources() -> std::shared_ptr<GNMTExtraResources>
    {
        return mGNMTExtraResourcesPool->Pop([](GNMTExtraResources* ptr) {});
    }

private:
    int mDevice{-1};
    int mBenchmarkBatchMin{1};
    int mBenchmarkBatchInc{1};
    std::string mBenchmarkInputFile;
    const std::chrono::microseconds mLatencyBound;

    // Resources and running parameters
    std::shared_ptr<InferenceManager> mResources;
    std::map<int, std::map<int, std::shared_ptr<GNMTInferRunner>>> mRunnersByBatchSizeAndSeqLength;
    std::map<int, std::map<int, std::shared_ptr<Model>>> mGeneratorPtrBySeqLengthAndBatchSize;
    int mMaxBatchSize{0};

    // GNMTExtraResources pool that holds extra memory besides trt buffer memory
    std::shared_ptr<Pool<GNMTExtraResources>> mGNMTExtraResourcesPool;
};

class Server : public mlperf::SystemUnderTest
{
public:
    //!
    //! \brief Initialize server resouces
    //!
    Server(std::string name, std::chrono::microseconds latency_bound, int num_gpus, bool isHomogeneousCluster) : mName{name}, mLatencyBound{latency_bound}, mNumGpus{num_gpus}, mHomogeneousCluster{isHomogeneousCluster}
    {
        const std::string cpuStrings = "0-" + std::to_string(num_gpus-1);
        const auto& batcher_thread_affinity = Affinity::GetCpusFromString(cpuStrings);
        mBatcherThreadPool = std::make_unique<ThreadPool>(batcher_thread_affinity);

        LOG(INFO) << "batcher threadPool size: " << mBatcherThreadPool->Size();

        // Set up single gpu server and queues; one thread per gpu
        for(int i = 0; i < mNumGpus; i++)
        {
            auto singleGpuServer = std::make_shared<SingleGPUServer>(i, mLatencyBound);
            gpuServers.push_back(singleGpuServer);

            work_queues.push_back(BlockingConcurrentQueue<WorkPacket>());
        }
    }

    ~Server() { }
    // SUT virtual interface
    virtual const std::string &Name() const { return mName; }
    virtual void IssueQuery(const std::vector<mlperf::QuerySample> &samples);
    virtual void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency> &latencies_ns);
    virtual void FlushQueries();

    //!
    //! \brief Launch single gpu exection by selecting SingleGPUServer based on device id
    //!
    void SingleGPUExecution(int device, int concurrency, std::vector<std::string> engine_dirs, std::shared_ptr<SampleLibrary> qsl, int FLAGS_timeout);
    
    const std::chrono::microseconds GetLatencyBound()
    {
        return mLatencyBound;
    }


    std::vector<BlockingConcurrentQueue<WorkPacket>> work_queues;
    std::vector<std::shared_ptr<SingleGPUServer>> gpuServers;
    volatile bool running{true};

private:
    // constants
    const std::string mName;
    const std::chrono::microseconds mLatencyBound;
    const int mNumGpus;
    const bool mHomogeneousCluster;
    
    int benchmarkCount{0};
    std::map<double, uint32_t> mBatchSizeByElapsedTime;
    std::map<uint32_t, double> mEstimatedExecutionTimeByBatchSize;

    int mQueueCount{0};

    // resources
    std::unique_ptr<ThreadPool> mBatcherThreadPool;
};
