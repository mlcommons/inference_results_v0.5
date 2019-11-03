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

#ifndef _INT4_OFFLINE_SAMPLE_LIBRARY_H
#define _INT4_OFFLINE_SAMPLE_LIBRARY_H

#include "query_sample_library.h"
#include <map>

class SampleLibrary : public mlperf::QuerySampleLibrary {
public:
  SampleLibrary(std::string name, std::string mapPath, std::string tensorPath, size_t perfSampleCount, void *imagePtr);
  ~SampleLibrary();

  virtual const std::string &Name() const { return m_Name; }
  virtual size_t TotalSampleCount() { return m_SampleCount; }
  virtual size_t PerformanceSampleCount() { return m_PerfSampleCount; }
  virtual void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples);
  virtual void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples);

private:
  const std::string m_Name;

  size_t m_SampleCount{ 0 };
  size_t m_PerfSampleCount{ 0 };

  // maps sampleId to <fileName, label>
  std::map<mlperf::QuerySampleIndex, std::tuple<std::string, size_t>> m_FileLabelMap;

  std::string m_TensorPath;
  std::string m_MapPath;
  void *m_ImagePtr;
};

#endif //_INT4_OFFLINE_SAMPLE_LIBRARY_H
