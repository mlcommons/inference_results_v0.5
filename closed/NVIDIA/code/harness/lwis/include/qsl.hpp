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

#ifndef __QSL_HPP__
#define __QSL_HPP__

#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <deque>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "logger.h"
#include <glog/logging.h>

#include "query_sample_library.h"
#include "test_settings.h"

// QSL (Query Sample Library) is an implementation of the MLPerf Query Sample Library.  It's purpose
// is to:
// 1) Allow samples to be loaded and unloaded dynamically at runtime from Loadgen.
// 2) Support lookup of currently loaded tensor addresses in memory.

namespace qsl {

  class SampleLibrary : public mlperf::QuerySampleLibrary {
  public:
    SampleLibrary(std::string name, std::string mapPath, std::string tensorPath, size_t perfSampleCount, size_t padding = 0) : m_Name(name), m_PerfSampleCount(perfSampleCount), m_PerfSamplePadding(padding), m_MapPath(mapPath), m_TensorPath(tensorPath) {
      // load and read in the sample map
      std::ifstream fs(m_MapPath);
      CHECK(fs) << "Unable to open sample map file: " << m_MapPath;

      char s[1024];
      while (fs.getline(s, 1024)) {
        std::istringstream iss(s);
        std::vector<std::string> r((std::istream_iterator<std::string>{iss}), std::istream_iterator<std::string>());

        m_FileLabelMap.insert(std::make_pair(m_SampleCount, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
        m_SampleCount++;
      }

      // as a safety, don't allow the perfSampleCount to be larger than sampleCount.
      m_PerfSampleCount = std::min(m_PerfSampleCount, m_SampleCount);
    }

    ~SampleLibrary() {
      CHECK_EQ(cudaFreeHost(m_HostSamples), cudaSuccess);
    }

    virtual const std::string &Name() const { return m_Name; }
    virtual size_t TotalSampleCount() { return m_SampleCount; }
    virtual size_t PerformanceSampleCount() { return m_PerfSampleCount; }

    virtual void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) {
      auto m = std::max_element(samples.begin(), samples.end());

      size_t sampleIndex = 0;
      // copy the samples into pinned memory
      for (auto &sampleId : samples) {
        std::string path = m_TensorPath + "/" + std::get<0>(m_FileLabelMap[sampleId]) + ".npy";
        std::vector<char> data;

        LoadNpyFile(data, path);

        if (m_HostSamples == nullptr) {
          // support buffering for the largest perf sample number
          CHECK_EQ(cudaMallocHost(&m_HostSamples, (m_PerfSampleCount + m_PerfSamplePadding) * m_SampleSizes[0]), cudaSuccess);

          // initialize free list
          for (int i = 0; i < m_PerfSampleCount + m_PerfSamplePadding; i++) m_SampleFreeList.push_back(static_cast<int8_t *>(m_HostSamples) + i * m_SampleSizes[0]);
        }

        //auto sampleAddress = AllocateSample(sampleId);
        auto sampleAddress = static_cast<int8_t *>(m_HostSamples) + sampleIndex++ * m_SampleSizes[0];
        m_SampleAddressMap[sampleId].push_back(sampleAddress);
        memcpy((char *)sampleAddress, &data[0], m_SampleSizes[0]);
      }
    }

    virtual void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) {
      // due to the removal of freelisting this code is currently a check and not required for functionality.
      for (auto &sampleId : samples) {
        auto it = m_SampleAddressMap.find(sampleId);
        CHECK(it != m_SampleAddressMap.end()) << "Sample: " << sampleId << " not allocated properly";
      
        auto &sampleAddresses = it->second;
        CHECK(!sampleAddresses.empty()) << "Sample: " << sampleId << " not loaded";

        //auto address = sampleAddresses.back();
        sampleAddresses.pop_back();
        // samples are now allocated in order and fully deallocated.  freelist no longer used.
        //m_SampleFreeList.push_back(address);
      
        if (sampleAddresses.empty()) m_SampleAddressMap.erase(it);
      }

      CHECK(m_SampleFreeList.size() == m_PerfSampleCount + m_PerfSamplePadding) << "FreeList size exceeds perf sample count";
      CHECK(m_SampleAddressMap.empty()) << "Unload did not remove all samples";
    }

    void *GetSampleAddress(mlperf::QuerySampleIndex sample_index) {
      auto it = m_SampleAddressMap.find(sample_index);
      CHECK(it != m_SampleAddressMap.end()) << "Sample: " << sample_index << " missing from RAM";
      return it->second.front();
    }

    size_t GetSampleSize() {
      return (m_SampleSizes.empty() ? 0 : m_SampleSizes.front());
    }

  private:
    void *AllocateSample(mlperf::QuerySampleIndex sampleId) {
      CHECK(0) << "AlocateSample should not be called";
      // NOTE: Due to the multi_stream scenario samples may be allocated multiple times and
      // so we ignore matches.  Assume that all other scenarios may move to this behavior.
      // This requires that all samples are unloaded.
      CHECK(!m_SampleFreeList.empty()) << "FreeList empty";
      auto address = m_SampleFreeList.front();
      m_SampleFreeList.pop_front();
      m_SampleAddressMap[sampleId].push_back(address);
      return address;
    }

    void LoadNpyFile(std::vector<char> &dst, std::string path) {
      std::ifstream fs(path, std::ios_base::in | std::ios_base::binary);
      CHECK(fs) << "Unable to open: " << path;
      char b[256];

      // magic and fixed header
      fs.read(b, 10);
      CHECK(fs) << "Unable to parse: " << path;

      // check magic
      CHECK(static_cast<unsigned char>(b[0]) == 0x93 && b[1] == 'N' && b[2] == 'U' && b[3] == 'M' && b[4] == 'P' && b[5] == 'Y') << "Bad magic: " << path;

      auto major = *reinterpret_cast<uint8_t *>(b + 6);
      auto minor = *reinterpret_cast<uint8_t *>(b + 7);
      auto headerSize = *reinterpret_cast<uint16_t *>(b + 8);
      fs.seekg(headerSize, std::ios_base::cur);

      if (m_SampleSizes.empty()) {
        // assume all tensor sizes are the same
        auto cur = fs.tellg();
        fs.seekg(0, std::ios::end);
        auto size = fs.tellg();
        m_SampleSizes.emplace_back(size - cur);
        fs.seekg(cur);
      }

      // assume the npy file stores the tensor in the correct format
      dst.resize(m_SampleSizes[0]);
      fs.read(&dst[0], m_SampleSizes[0]);
      CHECK(fs) << "Unable to parse: " << path;
      CHECK(fs.peek() == EOF) << "Did not consume full file: " << path;

      fs.close();
    }

    const std::string m_Name;
    // maps sampleId to <fileName, label>
    std::map<mlperf::QuerySampleIndex, std::tuple<std::string, size_t>> m_FileLabelMap;
    // maps sampleId to <address>
    std::map<mlperf::QuerySampleIndex, std::vector<void *>> m_SampleAddressMap;
    std::deque<void *> m_SampleFreeList;

    void *m_HostSamples{nullptr};
    std::vector<size_t> m_SampleSizes;
    nvinfer1::DataType m_Precision{nvinfer1::DataType::kFLOAT};

    size_t m_SampleCount{0};
    size_t m_PerfSampleCount{0};
    size_t m_PerfSamplePadding{0};
    std::string m_MapPath;
    std::string m_TensorPath;
  };

  typedef std::shared_ptr<qsl::SampleLibrary> SampleLibraryPtr_t;

};

#endif
