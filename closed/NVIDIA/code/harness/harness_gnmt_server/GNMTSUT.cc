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
#include "GNMTSUT.h"

#define likely(x) __builtin_expect((x), 1)

void SingleGPUServer::Setup(std::vector<std::string> engine_dirs, int concurrency)
{
    std::shared_ptr<Runtime> runtime = std::make_shared<StandardRuntime>();

    mGNMTExtraResourcesPool = Pool<GNMTExtraResources>::Create();
    std::vector<std::shared_ptr<GNMTExtraResources>> extraResources;
    
    extraResources.resize(concurrency);
    for(int i = 0; i < concurrency; i++)
    {
        extraResources[i] = std::make_shared<GNMTExtraResources>();
    }

    for(auto dir: engine_dirs)
    {
      // configurate the gnmt wrapper
      std::string jsonString = dir+"/"+"config.json";
      LOG(INFO) << jsonString;
      auto config = std::make_shared<Config>(jsonString);

      for(auto currentSeqLen: config->encoderMaxSeqLengths)
      {
          // create gnmt wrapper
          auto gnmt = std::make_shared<GNMTModel>(config, dir);

          int max_batch_size_per_gnmt{0};

          std::vector<string> engine_names;
          std::vector<std::shared_ptr<Model>> models;

          // set up gnmt
          gnmt->setup(runtime, mDevice, max_batch_size_per_gnmt, engine_names, models, currentSeqLen);

          mResources->RegisterModel(engine_names, models);

          mRunnersByBatchSizeAndSeqLength[max_batch_size_per_gnmt][currentSeqLen] = std::make_shared<GNMTInferRunner>(gnmt, mResources);

          mGeneratorPtrBySeqLengthAndBatchSize[currentSeqLen][max_batch_size_per_gnmt] = gnmt->GetGeneratorSmartPtr();

          mMaxBatchSize = std::max(max_batch_size_per_gnmt, mMaxBatchSize);
      }

      for(int i = 0; i < concurrency; i++)
      {
          extraResources[i]->registerConfig(config);
      }
    }

    LOG(INFO) << "Max batch size=" << mMaxBatchSize;

    LOG(INFO) << "Allocating Resources ... with device id: " << mDevice;

    mResources->AllocateResources();

    for(int i = 0; i < concurrency; i++)
    {
        extraResources[i]->allocateResources();
        mGNMTExtraResourcesPool->Push(extraResources[i]);
    }
}


void SingleGPUServer::BenchMark(std::map<double, uint32_t> &batch_size_by_elapsed_time, std::map<uint32_t, double> &estimated_execution_time_by_batch_size)
{
    if(mBenchmarkInputFile.empty())
    {
        return;
    }

    // load sentences for benchmark
    std::vector<std::string> enSentence;
    std::ifstream inputEnFile(mBenchmarkInputFile);
    if (!inputEnFile)
    {
        LOG(INFO) << "Error opening input file " << mBenchmarkInputFile;
        CHECK(false);
    }

    std::string line;
    while (std::getline(inputEnFile, line))
    {
        enSentence.push_back(line);
    }

    std::map<size_t, size_t> sequenceLengths;
    for (int index = 0; index < enSentence.size(); ++index)
    {
        sequenceLengths[index] = get_stringLength(enSentence[index]);
    }

    // test lstm benchmark
    for(int batch_size = mBenchmarkBatchMin; batch_size <= mMaxBatchSize; batch_size += mBenchmarkBatchInc)
    {
        InferBenGNMTBench benchmark(mResources, enSentence);
        auto runner = (mRunnersByBatchSizeAndSeqLength.lower_bound(batch_size)->second).begin()->second;
        auto model = runner->GetModelSmartPtr();

        // GNMT extra resources
        auto extraResources = getExtraResources();

        auto warmup = benchmark.Run(model, batch_size, extraResources, 0.2);
        auto result = benchmark.Run(model, batch_size, extraResources, 5.0);

        using namespace trtlab::TensorRT;
        auto time_per_batch = result->at(kExecutionTimePerBatch);
        estimated_execution_time_by_batch_size[batch_size] = time_per_batch;

        auto batching_window = std::chrono::duration<double>(mLatencyBound).count() - time_per_batch;

        if(batching_window <= 0.0)
        {
            LOG(INFO) << "Batch Sizes >= " << batch_size << " exceed latency threshold";
            break;
        }
        batch_size_by_elapsed_time[batching_window] = batch_size;
        mMaxBatchSize = std::max(mMaxBatchSize, batch_size);

        LOG(INFO) << "Batch Size: " << batch_size << " ExecutionTimePerBatch:  " << result->at(kExecutionTimePerBatch);
        LOG(INFO) << "Batch Size: " << batch_size << " MaxElapsedTime: " << batching_window;
    }

    batch_size_by_elapsed_time[std::numeric_limits<double>::max()] = mBenchmarkBatchMin;
}


void SingleGPUServer::Infer(std::vector<WorkPacket> &work_packets, std::shared_ptr<SampleLibrary> qsl, int running_batch_size)
{
    work_packets.resize(running_batch_size);

    // create BatchedTranslationTask from WorkPacket
    auto shared_wps =
        std::make_shared<std::vector<WorkPacket>>(std::move(work_packets));

    BatchedTranslationTask batch;
    batch.batchedStrings.resize(running_batch_size);
    batch.r_ids.resize(running_batch_size);

    int max_seq_length = 0;
    for (size_t i = 0; i < running_batch_size; i++)
    {
        batch.batchedStrings.at(i) = qsl->GetDataById((*shared_wps).at(i).sample.index);
        batch.r_ids.at(i) = (*shared_wps).at(i).sample.id;
        max_seq_length = std::max(max_seq_length, (int)(get_stringLength(batch.batchedStrings.at(i))));
    }

    // get runner
    auto gnmt_runner = (mRunnersByBatchSizeAndSeqLength.lower_bound(running_batch_size)->second).lower_bound(max_seq_length)->second;

    // TRT bindings
    auto gnmtBindings = std::make_shared<GNMTBindings>(gnmt_runner -> GetModelSmartPtr(), mGeneratorPtrBySeqLengthAndBatchSize.lower_bound(max_seq_length)->second);
    // GNMT extra resources
    auto extraResources = getExtraResources();

    gnmtBindings->createBindingsAndExecutionContext(mResources, extraResources, running_batch_size);

    gnmt_runner->Infer(
        batch,

        gnmtBindings,

        [this, wps = shared_wps](std::shared_ptr<GNMTBindings> bindings) {

        }

        // TO DO: something with the deadline
        );

    // reset work_packets
    CHECK_EQ(work_packets.size(), 0);
    work_packets.resize(mMaxBatchSize);
}

void Server::IssueQuery(const std::vector<mlperf::QuerySample> &samples)
{
    for(auto sample : samples)
    {
        work_queues[0].enqueue(WorkPacket{sample,
                                          std::chrono::high_resolution_clock::now()});
        mQueueCount++;
        mQueueCount %= mNumGpus;
    }
}

void Server::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency> &latencies_ns)
{
    LOG(INFO) << "Report Latency Results";

    std::vector<float> float_latencies;
    float_latencies.resize(0);

    for(auto latency_ns : latencies_ns)
    {
        if(latency_ns > 0)
        {
            float lantency_ms = (float)(latency_ns * 1e-6);
            float_latencies.push_back(lantency_ms);
        }
    }

    float average = accumulate( float_latencies.begin(), float_latencies.end(), 0.0)/float_latencies.size();
    LOG(INFO) << "Average latency " << average << "ms";

    LOG(INFO) << "50 percentile latency " << percentile(float_latencies, 50) << "ms";

    LOG(INFO) << "90 percentile latency " << percentile(float_latencies, 90) << "ms";

    LOG(INFO) << "97 percentile latency " << percentile(float_latencies, 97) << "ms";

}

void Server::FlushQueries()
{

}

void Server::SingleGPUExecution(int device, int concurrency, std::vector<std::string> engine_dirs, std::shared_ptr<SampleLibrary> qsl, int FLAGS_timeout)
{
    if(device >= mBatcherThreadPool->Size())
    {
        LOG(INFO) << "Out of gpu range";
        return;
    }

    auto resources = std::make_shared<InferenceManager>(concurrency, concurrency+4);

    // loading gnmt engines
    LOG(INFO) << "GNMT Seting up ... with device id: " << device;
    gpuServers[device]->Initialize(resources);
    gpuServers[device]->Setup(engine_dirs, concurrency);

    // Batching timeout
    constexpr uint64_t quanta = 5; // microseconds
    const double timeout = static_cast<double>(FLAGS_timeout - quanta) / 1000000.0; // microseconds

    // Main Loop
    mBatcherThreadPool->enqueue([this, qsl, gpuServer = gpuServers[device], device, timeout]() mutable {

        size_t total_count;
        size_t max_deque;
        size_t adjustable_max_batch_size;
        int max_batch_size = gpuServer->getMaxBatchSize();

        std::vector<WorkPacket> work_packets(max_batch_size);

        thread_local ConsumerToken token(this->work_queues[0]);
        double elapsed, elapsed_timeout;
        double quanta_in_secs = quanta / 1000000.0; // convert microsecs to
        const double latency_budget = std::chrono::duration<double>(this->GetLatencyBound()).count();
        auto elapsed_time = [](std::chrono::high_resolution_clock::time_point start) -> double {
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start)
                .count();
        };

        CHECK_EQ(cudaSetDevice(device), CUDA_SUCCESS) << "fail to launch device " << 1;

        // Batching Loop
        for(; likely(this->running);)
        {
            total_count = 0;
            max_deque = max_batch_size;
            // if we have a work packet, we stay in this loop and continue to batch until the
            // timeout is reached
            auto count = (this->work_queues[0]).wait_dequeue_bulk_timed(token, &work_packets[total_count],
                                                            max_deque, quanta);
            total_count = count;
            
            // batching complete, now queue the execution
            if(total_count)
            {   
                gpuServer->Infer(work_packets, qsl, total_count);
            }

        }
    });
}
