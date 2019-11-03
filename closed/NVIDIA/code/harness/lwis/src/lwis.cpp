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

#include "lwis.hpp"
#include "loadgen.h"
#include "query_sample_library.h"
#include "lwis_buffers.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

#include <fstream>
#include <algorithm>

#include "logger.h"
#include <glog/logging.h>

namespace lwis {
  using namespace std::chrono_literals;

  std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
      res.push_back(item);
    }
    return res;
  }

  //----------------
  // Device
  //----------------
  void Device::AddEngine(EnginePtr_t engine) {
    size_t batchSize = engine->GetCudaEngine()->getMaxBatchSize();

    m_Engines[batchSize].emplace_back(engine);
    m_BatchSize = std::min(m_BatchSize, batchSize);
  }

  void Device::BuildGraphs() {
    Issue();

    size_t batchSize = 1;
    for (auto &e : m_Engines) {
      auto maxBatchSize = e.first;
      auto engine = e.second;

      // build the graph by performing a single execution.  the engines are stored in ascending
      // order of maxBatchSize.  build graphs up to and including this size
      while (batchSize <= maxBatchSize) {
        for (auto &streamState : m_StreamState) {
          auto &stream = streamState.first;
          auto &state = streamState.second;
          auto &context = std::get<4>(state);

          auto bufferManager = std::get<0>(state);

          cudaGraph_t graph;
#if (CUDA_VERSION >= 10010)
          CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0], cudaStreamCaptureModeRelaxed), CUDA_SUCCESS);
#else
          CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0]), CUDA_SUCCESS);
#endif
          CHECK_EQ(context->enqueue(batchSize, &(bufferManager->getDeviceBindings()[0]), m_InferStreams[0], nullptr), true);
          CHECK_EQ(cudaStreamEndCapture(m_InferStreams[0], &graph), CUDA_SUCCESS);

          t_GraphKey key = std::make_pair(stream, batchSize);
          m_CudaGraphs[key] = graph;

          cudaGraphExec_t graphExec;
          CHECK_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), CUDA_SUCCESS);
          m_CudaGraphExecs[key] = graphExec;
        }

        batchSize++;
      }
    }
  }

  void Device::Setup() {
    cudaSetDevice(m_Id);
    if (m_EnableDeviceScheduleSpin) cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    unsigned int cudaEventFlags = (m_EnableSpinWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming;

    for (auto &stream : m_InferStreams) {
      CHECK_EQ(cudaStreamCreate(&stream), CUDA_SUCCESS);
    }

    for (auto &stream : m_CopyStreams) {
      CHECK_EQ(cudaStreamCreate(&stream), CUDA_SUCCESS);

      std::shared_ptr<nvinfer1::ICudaEngine> emptyPtr{};
      std::shared_ptr<nvinfer1::ICudaEngine> aliasPtr(emptyPtr, m_Engines.rbegin()->second.back()->GetCudaEngine());
      auto state = std::make_tuple(std::make_shared<BufferManager>(aliasPtr, m_BatchSize), cudaEvent_t(), cudaEvent_t(), cudaEvent_t(), m_Engines.rbegin()->second.back()->GetCudaEngine()->createExecutionContext());
      CHECK_EQ(cudaEventCreateWithFlags(&std::get<1>(state), cudaEventFlags), CUDA_SUCCESS);
      CHECK_EQ(cudaEventCreateWithFlags(&std::get<2>(state), cudaEventFlags), CUDA_SUCCESS);
      CHECK_EQ(cudaEventCreateWithFlags(&std::get<3>(state), cudaEventFlags), CUDA_SUCCESS);

      m_StreamState.insert(std::make_pair(stream, state));

      m_StreamQueue.emplace_back(stream);
    }
  }

  void Device::Issue() {
    CHECK_EQ(cudaSetDevice(m_Id), cudaSuccess);
  }

  void Device::Done() {
    // join before destroying all members
    m_Thread.join();

    // destroy member objects
    cudaSetDevice(m_Id);

    for (auto &stream : m_InferStreams) {
      cudaStreamDestroy(stream);
    }
    for (auto &stream : m_CopyStreams) {
      auto &state = m_StreamState[stream];

      cudaStreamDestroy(stream);
      cudaEventDestroy(std::get<1>(state));
      cudaEventDestroy(std::get<2>(state));
      cudaEventDestroy(std::get<3>(state));
      std::get<4>(state)->destroy();
    }
  }

  void Device::Completion() {
    // Testing for completion needs to be based on the main thread finishing submission and
    // providing events for the completion thread to wait on.  The resources exist as part of the
    // Device class.
    //
    // Samples and responses are assumed to be contiguous.

    // Flow:
    // Main thread
    // - Find Device (check may be based on data buffer availability)
    // - Enqueue work
    // - Enqueue CompletionQueue batch
    // ...
    // - Enqueue CompletionQueue null batch

    // Completion thread(s)
    // - Wait for entry
    // - Wait for queue head to have data ready (wait for event)
    // - Dequeue CompletionQueue

    while (true) {
      // TODO: with multiple CudaStream inference it may be beneficial to handle these out of order
      auto batch = m_CompletionQueue.front();

      if (batch.Responses.empty()) break;

      // wait on event completion
      CHECK_EQ(cudaEventSynchronize(batch.Event), cudaSuccess);

      // callback if it exists
      if (m_ResponseCallback)
      {
        CHECK(batch.SampleIds.size() == batch.Responses.size()) << "missing sample IDs";
        m_ResponseCallback(&batch.Responses[0], batch.SampleIds, batch.Responses.size());
      }

      // assume this function is reentrant for multiple devices
      mlperf::QuerySamplesComplete(&batch.Responses[0], batch.Responses.size());

      m_StreamQueue.emplace_back(batch.Stream);
      m_CompletionQueue.pop_front();
    }
  }

  //----------------
  // Server
  //----------------

  //! Setup
  //!
  //! Perform all necessary (untimed) setup in order to perform inference including: building
  //! graphs and allocating device memory.
  void Server::Setup(ServerSettings &settings, ServerParams &params) {
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    m_ServerSettings = settings;

    // enumerate devices
    std::vector<size_t> devices;
    if (params.DeviceNames == "all") {
      int numDevices = 0;
      cudaGetDeviceCount(&numDevices);
      for (int i = 0; i < numDevices; i++) {
        devices.emplace_back(i);
      }
    }
    else {
      auto deviceNames = split(params.DeviceNames, ',');
      for (auto &n : deviceNames) devices.emplace_back(std::stoi(n));
    }

    // check if an engine was specified
    if (!params.EngineNames.size()) gLogError << "Engine file(s) not specified" << std::endl;

    auto runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

    for (auto &deviceNum : devices) {
      cudaSetDevice(deviceNum);

      size_t type = 0;
      for (auto &deviceTypes : params.EngineNames) {

        for (auto &batches : deviceTypes) {
          auto isDlaDevice = type == 1;

          if (!isDlaDevice && m_ServerSettings.MaxGPUs != -1 && deviceNum >= m_ServerSettings.MaxGPUs) continue;

          for (uint32_t deviceInstance = 0; deviceInstance < (isDlaDevice ? runtime->getNbDLACores() : 1); deviceInstance++) {
            if (isDlaDevice && m_ServerSettings.MaxDLAs != -1 && deviceInstance >= m_ServerSettings.MaxDLAs) continue;

            size_t numCopyStreams = isDlaDevice ? m_ServerSettings.DLACopyStreams : m_ServerSettings.GPUCopyStreams;
            size_t numInferStreams = isDlaDevice ? m_ServerSettings.DLAInferStreams : m_ServerSettings.GPUInferStreams;
            size_t batchSize = isDlaDevice ? m_ServerSettings.DLABatchSize : m_ServerSettings.GPUBatchSize;

            auto device = std::make_shared<lwis::Device>(deviceNum, numCopyStreams, numInferStreams, m_ServerSettings.EnableSpinWait, m_ServerSettings.EnableDeviceScheduleSpin, batchSize, isDlaDevice);
            m_Devices.emplace_back(device);

            for (auto &engineName : batches) {
              std::vector<char> trtModelStream;
              auto size = GetModelStream(trtModelStream, engineName);
              if (isDlaDevice) runtime->setDLACore(deviceInstance);
              auto engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
              auto batchSize = engine->getMaxBatchSize();

              device->AddEngine(std::make_shared<lwis::Engine>(engine));

              std::ostringstream deviceName;
              deviceName << "Device:" << deviceNum;
              if (isDlaDevice) { deviceName << ".DLA-" << deviceInstance; }

              device->m_Name = deviceName.str();
              gLogInfo << device->m_Name << ": " << engineName << " has been successfully loaded." << std::endl;
            }
          }
        }

        type++;
      }
    }

    runtime->destroy();

    CHECK(m_Devices.size()) << "No devices or engines available";

    for (auto &device : m_Devices) device->Setup();

    if (m_ServerSettings.EnableCudaGraphs) for (auto &device : m_Devices) if (!device->m_DLA) device->BuildGraphs();

    Reset();

    // create batchers
    for (size_t deviceNum = 0; deviceNum < (m_ServerSettings.EnableBatcherThreadPerDevice ? m_Devices.size() : 1); deviceNum++) {
      gLogInfo << "Creating batcher thread: " << deviceNum << " EnableBatcherThreadPerDevice: " << (m_ServerSettings.EnableBatcherThreadPerDevice ? "true" : "false") << std::endl;
      m_Threads.emplace_back(std::thread(&Server::ProcessSamples, this));
    }

    // create issue threads
    if (m_ServerSettings.EnableCudaThreadPerDevice) {
      for (size_t deviceNum = 0; deviceNum < m_Devices.size(); deviceNum++) {
        gLogInfo << "Creating cuda thread: " << deviceNum << std::endl;
        m_IssueThreads.emplace_back(std::thread(&Server::ProcessBatches, this));
      }
    }
  }

  void Server::Done() {
    // send dummy batch to signal completion
    for (auto &device : m_Devices) device->m_CompletionQueue.push_back(Batch{});
    for (auto &device : m_Devices) device->Done();

    // send end sample to signal completion
    while (!m_WorkQueue.empty()) { }

    while (m_DeviceNum) {
      size_t currentDeviceId = m_DeviceNum;
      m_WorkQueue.emplace_back(mlperf::QuerySample{0, 0});
      while (currentDeviceId == m_DeviceNum) { }
    }

    if (m_ServerSettings.EnableCudaThreadPerDevice) {
      for (auto &device : m_Devices) {
        std::deque<mlperf::QuerySample> batch;
        auto pair = std::make_pair(std::move(batch), nullptr);
        device->m_IssueQueue.emplace_back(pair);
      }
      for (auto &thread : m_IssueThreads) thread.join();
    }
      
    // join after we insert the dummy sample
    for (auto &thread : m_Threads) thread.join();
  }

  void Server::IssueQuery(const std::vector<mlperf::QuerySample> &samples) {
    m_WorkQueue.insert(samples);
  }

  DevicePtr_t Server::GetNextAvailableDevice(size_t deviceId) {
    DevicePtr_t device;
    if (!m_ServerSettings.EnableBatcherThreadPerDevice) {
      do {
        device = m_Devices[m_DeviceIndex];
        m_DeviceIndex = (m_DeviceIndex + 1) % m_Devices.size();
      } while (device->m_StreamQueue.empty());
    }
    else {
      device = m_Devices[deviceId];
      while (device->m_StreamQueue.empty()) { }
    }

    return device;
  }

  void Server::IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream) {
    auto enqueueBatchSize = device->m_DLA ? device->m_BatchSize : batchSize;
    auto inferStream = (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : device->m_InferStreams[device->m_InferStreamNum];

    auto &state = device->m_StreamState[copyStream];
    auto bufferManager = std::get<0>(state);
    auto &htod = std::get<1>(state);
    auto &inf = std::get<2>(state);
    auto &dtoh = std::get<3>(state);
    auto &context = std::get<4>(state);

    // setup Device
    device->Issue();

    // perform copy to device
#ifndef LWIS_DEBUG_DISABLE_INFERENCE
    std::vector<void *> buffers = bufferManager->getDeviceBindings();
    if (m_ServerSettings.EnableDma) buffers = CopySamples(device, batchSize, begin, end, copyStream, device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess || m_ServerSettings.EnableDirectHostAccess, m_ServerSettings.EnableDmaStaging);
    if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);

#ifndef LWIS_DEBUG_DISABLE_COMPUTE
    // perform inference
    if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
    Device::t_GraphKey key = std::make_pair(copyStream, enqueueBatchSize);
    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
    if (g_it != device->m_CudaGraphExecs.end()) {
      CHECK_EQ(cudaGraphLaunch(g_it->second, inferStream), CUDA_SUCCESS);
    }
    else {
      CHECK_EQ(context->enqueue(enqueueBatchSize, &buffers[0], inferStream, nullptr), true);
    }
#endif
    if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);

    // perform copy back to host
    if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
    if (m_ServerSettings.EnableDma && (!device->m_DLA || !m_ServerSettings.EnableDLADirectHostAccess) && !m_ServerSettings.EnableDirectHostAccess) bufferManager->copyOutputToHostAsync(copyStream);
#endif
    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);

    // optional synchronization
    if (m_ServerSettings.EnableSyncOnEvent) cudaEventSynchronize(dtoh);

    // generate asynchronous response
    if (m_ServerSettings.EnableResponse) {
      // FIXME: support multiple inputs and outputs
      auto buffer = static_cast<int8_t *>(bufferManager->getHostBuffer(1));
      size_t sampleSize = volume(device->m_Engines.rbegin()->second.back()->GetCudaEngine()->getBindingDimensions(1)) * getElementSize(device->m_Engines.rbegin()->second.back()->GetCudaEngine()->getBindingDataType(1));

      Batch batch;
      for (auto it = begin; it != end; ++it) {
        batch.Responses.emplace_back(mlperf::QuerySampleResponse{it->id, (uintptr_t)buffer, sampleSize});
        if (device->m_ResponseCallback) batch.SampleIds.emplace_back(it->index);
        buffer += sampleSize;
      }

      batch.Event = dtoh;
      batch.Stream = copyStream;

      device->m_CompletionQueue.emplace_back(batch);
    }

    // Simple round-robin across inference streams.  These don't need to be managed like copy
    // streams since they are not tied with a resource that is re-used and not managed by hardware.
    device->m_InferStreamNum = (device->m_InferStreamNum + 1) % device->m_InferStreams.size();

    if (device->m_Stats.m_BatchSizeHistogram.find(batchSize) == device->m_Stats.m_BatchSizeHistogram.end()) {
      device->m_Stats.m_BatchSizeHistogram[batchSize] = 1;
    } else {
      device->m_Stats.m_BatchSizeHistogram[batchSize]++;
    }
  }

  std::vector<void *> Server::CopySamples(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, cudaStream_t stream, bool directHostAccess, bool staging) {
    // Cover the following conditions:
    // 1) No sample library.  This is a debug mode and will copy whatever data is in the host buffer.
    // 2) Unified memory.  If contiguous supply the pointer directly to the engine (no copy).  Else
    // copy into the device buffer.

    auto bufferManager = std::get<0>(device->m_StreamState[stream]);

    // setup default device buffers based on modes
    std::vector<void *> buffers;
    if (directHostAccess) {
      buffers = bufferManager->getHostBindings();
    }
    else {
      buffers = bufferManager->getDeviceBindings();
    }

    if (m_SampleLibrary) {
      // detect contiguous samples
      bool contiguous = true;
      auto prev = static_cast<int8_t *>(m_SampleLibrary->GetSampleAddress(begin->index));

      // test sample size vs buffer size derived from engine
      size_t sampleSize = volume(device->m_Engines.rbegin()->second.front()->GetCudaEngine()->getBindingDimensions(0),
                                 device->m_Engines.rbegin()->second.front()->GetCudaEngine()->getBindingFormat(0))
        * getElementSize(device->m_Engines.rbegin()->second.front()->GetCudaEngine()->getBindingDataType(0));
      CHECK(m_SampleLibrary->GetSampleSize() == sampleSize) << "Sample size (" << m_SampleLibrary->GetSampleSize()
                                                            << ") does not match engine input size ("
                                                            << sampleSize << ")";

      if (!m_ServerSettings.ForceContiguous) {
        for (auto it = begin + 1; it != end; ++it) {
          auto next = static_cast<int8_t *>(m_SampleLibrary->GetSampleAddress(it->index));
          if (next != prev + m_SampleLibrary->GetSampleSize()) {
            contiguous = false;
            break;
          }
          
          prev = next;
        }
      }

      if (!contiguous) {
        size_t offset = 0;
        for (auto it = begin; it != end; ++it) {
          if (directHostAccess) {
            // copy to the host staging buffer which is used as device buffer
            memcpy(static_cast<int8_t *>(bufferManager->getHostBuffer(0)) + offset++ * m_SampleLibrary->GetSampleSize(), m_SampleLibrary->GetSampleAddress(it->index), m_SampleLibrary->GetSampleSize());
            device->m_Stats.m_MemcpyCalls++;
          }
          else if (staging) {
            // copy to the host staging buffer and then to device buffer
            memcpy(static_cast<int8_t *>(bufferManager->getHostBuffer(0)) + offset++ * m_SampleLibrary->GetSampleSize(), m_SampleLibrary->GetSampleAddress(it->index), m_SampleLibrary->GetSampleSize());
            bufferManager->copyInputToDeviceAsync(stream);
            device->m_Stats.m_MemcpyCalls++;
          }
          else {
            // copy direct to device buffer
            bufferManager->copyInputToDeviceAsync(stream, m_SampleLibrary->GetSampleAddress(it->index), m_SampleLibrary->GetSampleSize(), offset++);
            device->m_Stats.m_PerSampleCudaMemcpyCalls++;
          }
        }
      }
      else {
        if (directHostAccess) {
          // access samples directly when they are contiguous
          buffers[0] = m_SampleLibrary->GetSampleAddress(begin->index);
        }
        else {
          // copy direct to device buffer with single DMA
          bufferManager->copyInputToDeviceAsync(stream, m_SampleLibrary->GetSampleAddress(begin->index), m_SampleLibrary->GetSampleSize() * batchSize);
          device->m_Stats.m_BatchedCudaMemcpyCalls++;
        }
      }
    }
    else {
      // no sample library.  copy to device memory if necessary
      if (!directHostAccess) {
        bufferManager->copyInputToDeviceAsync(stream);
        device->m_Stats.m_BatchedCudaMemcpyCalls++;
      }
    }

    return buffers;
  }

  void Server::Reset() {
    m_DeviceIndex = 0;

    for (auto &device: m_Devices) {
        device->m_InferStreamNum = 0;
        device->m_Stats.reset();
    }
  }

  void Server::ProcessSamples() {
    // until Setup is called we may not have valid devices
    size_t deviceId = m_DeviceNum++;

    // initial device available
    auto device = GetNextAvailableDevice(deviceId);

    while (true) {
      std::deque<mlperf::QuerySample> samples;
      do {
        m_WorkQueue.acquire(samples, m_ServerSettings.Timeout, device->m_BatchSize, m_ServerSettings.EnableDequeLimit || m_ServerSettings.EnableBatcherThreadPerDevice);
      } while(samples.empty());

      auto begin = samples.begin();
      auto end = samples.end();

      // Use a null (0) id to represent the end of samples
      if (!begin->id) {
        m_DeviceNum--;
        break;
      }

      auto batch_begin = begin;

      // build batches up to maximum supported batchSize
      while (batch_begin != end) {
        auto batchSize = std::min(device->m_BatchSize, static_cast<size_t>(std::distance(batch_begin, end)));
        auto batch_end = batch_begin + batchSize;

        // Acquire resources
        auto copyStream = device->m_StreamQueue.front();
        device->m_StreamQueue.pop_front();

        // Issue this batch
        if (!m_ServerSettings.EnableCudaThreadPerDevice) {
          // issue on this thread
          IssueBatch(device, batchSize, batch_begin, batch_end, copyStream);
        }
        else {
          // issue on device specific thread
          std::deque<mlperf::QuerySample> batch(batch_begin, batch_end);
          auto pair = std::make_pair(std::move(batch), copyStream);
          device->m_IssueQueue.emplace_back(pair);
        }

        // Advance to next batch
        batch_begin = batch_end;

        // Get available device for next batch
        device = GetNextAvailableDevice(deviceId);
      }
    }
  }

  void Server::ProcessBatches() {
    // until Setup is called we may not have valid devices
    size_t deviceId = m_IssueNum++;
    auto &device = m_Devices[deviceId];
    auto &issueQueue = device->m_IssueQueue;

    while (true) {
      auto pair = issueQueue.front();
      issueQueue.pop_front();

      auto &batch = pair.first;
      auto &stream = pair.second;

      if (batch.empty()) {
        m_IssueNum--;
        break;
      }

      IssueBatch(device, batch.size(), batch.begin(), batch.end(), stream);
    }
  }

  void Server::Warmup(double duration) {
    double elapsed = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    do {
      for (size_t deviceIndex = 0; deviceIndex < m_Devices.size(); ++deviceIndex) {
        // get next device to send batch to
        auto device = m_Devices[deviceIndex];

        for (auto copyStream : device->m_CopyStreams) {
          for (auto inferStream : device->m_InferStreams) {
            auto &state = device->m_StreamState[copyStream];
            auto bufferManager = std::get<0>(state);
            auto &htod = std::get<1>(state);
            auto &inf = std::get<2>(state);
            auto &dtoh = std::get<3>(state);
            auto &context = std::get<4>(state);

            device->Issue();

            if (m_ServerSettings.EnableDma && !(device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess) && !m_ServerSettings.EnableDirectHostAccess && !m_ServerSettings.EnableDmaStaging) {
              bufferManager->copyInputToDeviceAsync(copyStream);
            }
            if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);

            if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
            Device::t_GraphKey key = std::make_pair(inferStream, device->m_BatchSize);
            auto g_it = device->m_CudaGraphExecs.lower_bound(key);
            if (g_it != device->m_CudaGraphExecs.end()) {
              CHECK_EQ(cudaGraphLaunch(g_it->second, (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream), CUDA_SUCCESS);
            }
            else {
              auto buffers = bufferManager->getDeviceBindings();
              if (m_ServerSettings.EnableDLADirectHostAccess && device->m_DLA || m_ServerSettings.EnableDirectHostAccess) {
                buffers = bufferManager->getHostBindings();
              }
              CHECK_EQ(context->enqueue(device->m_BatchSize, &buffers[0], (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream, nullptr), true);
            }
            if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);

            if (!m_ServerSettings.RunInferOnCopyStreams) CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
            if (m_ServerSettings.EnableDma && !(device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess) && !m_ServerSettings.EnableDirectHostAccess && !m_ServerSettings.EnableDmaStaging) {
              bufferManager->copyOutputToHostAsync(copyStream);
            }
            CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);
          }
        }
      }
      elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);

    for (auto &device : m_Devices) {
      device->Issue();
      cudaDeviceSynchronize();
    }

    // reset server state
    Reset();
  }

  void Server::FlushQueries() {
    // This function is called at the end of a series of IssueQuery calls (typically the end of a
    // region of queries that define a performance or accuracy test).  Its purpose is to allow a
    // SUT to force all remaining queued samples out to avoid implementing timeouts.

    // Currently, there is no use case for it in this IS.
  }

  void Server::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency> &latencies_ns) {
  }

  void Server::SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex> &sample_ids, size_t response_count)> callback)
  {
    std::for_each(m_Devices.begin(), m_Devices.end(), [callback](DevicePtr_t device) { device->SetResponseCallback(callback); });
  }

};
