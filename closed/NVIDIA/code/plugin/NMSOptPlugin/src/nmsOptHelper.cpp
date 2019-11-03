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

#include "ssdOpt.h"
#include "ssdOptMacros.h"
#include <cassert>
#include <algorithm>

void reportAssertion(const char* msg, const char* file, int line)
{
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    cudaDeviceReset();
    abort();
}


namespace nvinfer1
{
namespace plugin
{

size_t detectionForwardBBoxDataSize(int N,
                                    int C1,
                                    DType_t DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return N * C1 * sizeof(float);
    }
    
    printf("Only FP32 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       DType_t DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return shareLocation ? 0 : N * C1 * sizeof(float);
    }
    printf("Only FP32 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardPreNMSSize(int N,
                                  int C2)
{
    static_assert(sizeof(float) == sizeof(int), "Must run on a platform where sizeof(int) == sizeof(float)");
    return N * C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int N,
                                   int numClasses,
                                   int topK)
{
    static_assert(sizeof(float) == sizeof(int), "Must run on a platform where sizeof(int) == sizeof(float)");
    return N * numClasses * topK * sizeof(float);
}

size_t detectionInferenceWorkspaceSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       int C2,
                                       int numClasses,
                                       int numPredsPerClass,
                                       int topK,
                                       DType_t DT_BBOX,
                                       DType_t DT_SCORE)
{
    size_t wss[9];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX); //bboxData
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX); //bboxPermute
    wss[2] = detectionForwardPreNMSSize(N, C2); //scores
    wss[3] = detectionForwardPreNMSSize(N, C2); //softmaxScores
    wss[4] = detectionForwardPreNMSSize(N, C2); //indices
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK); //postNMSScores
    wss[6] = detectionForwardPostNMSSize(N, numClasses, topK); //postNMSIndices
    wss[7] = N * numClasses * sizeof(int) + N * sizeof(int); //activeCount, activeCountPerBatch
    wss[8] = N * numClasses * numPredsPerClass * sizeof(float); //sortingWorkspace
    return calculateTotalWorkspaceSize(wss, 9);
}
} //namespace plugin
} //namespace nvinfer1
