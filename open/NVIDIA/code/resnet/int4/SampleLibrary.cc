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

#include "SampleLibrary.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <deque>
#include <vector>

bool loadFileToMemory(const char *path, void *buffer, int length) {
  FILE    *file=fopen(path, "r");
  uint8_t  *local=(uint8_t *)buffer;
 
  // used to parse int8 .npy files 
  if(file==NULL)
  {
    printf("File %s couldn't be opened\n", path);
    return false;
  }
    
  fseek(file, 128, SEEK_SET);
  int bytesRead = fread(local, 1, length, file);
  if(bytesRead!=length) {
    // short read
    fclose(file);
    return false;
  }
  fclose(file);
  return true;
}


//----------------
// QuerySampleLibrary
//----------------
SampleLibrary::SampleLibrary(std::string name, std::string mapPath, std::string tensorPath, size_t perfSampleCount, void *imagePtr) : m_Name(name), m_TensorPath(tensorPath), m_MapPath(mapPath), m_PerfSampleCount(perfSampleCount), m_ImagePtr(imagePtr)
{
  std::ifstream fs(m_MapPath);
  if(!fs.is_open()) 
  {
    printf("FATAL ERRORL Unable to open sample map file: %s \n", m_MapPath.c_str());
    exit(1);
  }

  char s[1024];
  while (fs.getline(s, 1024)) 
  {
    std::istringstream iss(s);
    std::vector<std::string> r((std::istream_iterator<std::string>{iss}), std::istream_iterator<std::string>());

    m_FileLabelMap.insert(std::make_pair(m_SampleCount, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
    m_SampleCount++;
  }

  // as a safety, don't allow the perfSampleCount to be larger than sampleCount.
  m_PerfSampleCount = std::min(m_PerfSampleCount, m_SampleCount);
}

SampleLibrary::~SampleLibrary() {
}

void SampleLibrary::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) {
  int32_t     imageSize {3 * 224 * 224};
 
  int i = 0;
  for (auto& sample : samples) 
  {
    std::string path = m_TensorPath + "/" + std::get<0>(m_FileLabelMap[sample]) + ".npy";

    //printf("File_path = %s\n", path.c_str());

    if(!loadFileToMemory(path.c_str(), m_ImagePtr + i * imageSize, imageSize))
    {
      printf("FATAL ERROR: Loading image '%s' failed\n", path.c_str());
      exit(1);      
    }
    i++;
 }
}

void SampleLibrary::UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) {
}
