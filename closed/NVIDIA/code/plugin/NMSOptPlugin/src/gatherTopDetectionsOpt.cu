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

#include <vector>

#include "ssdOpt.h"
#include "ssdOptMacros.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T_BBOX, typename T_SCORE, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void gatherTopDetectionsOpt_kernel(
        const bool shareLocation,
        const int numImages,
        const int numPredsPerClass,
        const int numClasses,
        const int topK,
        const int keepTopK,
        const int* indices,
        const T_SCORE* scores,
        const T_BBOX* bboxData,
        T_BBOX* topDetections)
{
    assert(keepTopK <= topK);

    //topDetections contains numImages continuous segments where each segment records
    // 1. 200 detections containing information of image id, label, confidence score and bboxes
    // 2. keepCount of that image that tells the number of valid detections out of 200

    for (int i = blockIdx.x * nthds_per_cta + threadIdx.x;
         i < numImages * keepTopK;
         i += gridDim.x * nthds_per_cta)
    {
        const int imgId = i / keepTopK;
        const int detId = i % keepTopK;
        const int imgBase = imgId * (7 * keepTopK + 1);
        const int offset = imgId * numClasses * topK;
        const int index = indices[offset + detId];
        const T_SCORE score = scores[offset + detId];
        if (index == -1)
        {
            topDetections[imgBase + detId * 7] = imgId;  // image id
            topDetections[imgBase + detId * 7 + 1] = 0;  // bbox ymin
            topDetections[imgBase + detId * 7 + 2] = 0;  // bbox xmin
            topDetections[imgBase + detId * 7 + 3] = 0;  // bbox ymax
            topDetections[imgBase + detId * 7 + 4] = 0;  // bbox xmax
            topDetections[imgBase + detId * 7 + 5] = 0;  // confidence score
            // score==0 will not pass the VisualizeBBox check
            topDetections[imgBase + detId * 7 + 6] = -1;  // label
        }
        else
        {
            const int bboxOffset = imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
            const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                        : index % (numClasses * numPredsPerClass)) + bboxOffset) * 4;
            topDetections[imgBase + detId * 7] = imgId;                                                            // image id
            // clipped bbox ymin
            topDetections[imgBase + detId * 7 + 1] = max(min(bboxData[bboxId + 1], T_BBOX(1.)), T_BBOX(0.));
            // clipped bbox xmin
            topDetections[imgBase + detId * 7 + 2] = max(min(bboxData[bboxId], T_BBOX(1.)), T_BBOX(0.));
            // clipped bbox ymax
            topDetections[imgBase + detId * 7 + 3] = max(min(bboxData[bboxId + 3], T_BBOX(1.)), T_BBOX(0.));
            // clipped bbox xmax
            topDetections[imgBase + detId * 7 + 4] = max(min(bboxData[bboxId + 2], T_BBOX(1.)), T_BBOX(0.));
            topDetections[imgBase + detId * 7 + 5] = score;                                                        // confidence score
            topDetections[imgBase + detId * 7 + 6] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass; // label
            atomicAdd(&((int*)topDetections)[imgBase + 7 * keepTopK], 1);
        }
    }
}

template <typename T_BBOX, typename T_SCORE>
ssdStatus_t gatherTopDetectionsOpt_gpu(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* topDetections)
{
    cudaMemsetAsync(topDetections, 0, numImages * (7 * keepTopK + 1) * sizeof(float), stream);
    const int BS = 32;
    const int GS = 32;
    gatherTopDetectionsOpt_kernel<T_BBOX, T_SCORE, BS><<<GS, BS, 0, stream>>>(shareLocation, numImages, numPredsPerClass,
                                                                           numClasses, topK, keepTopK,
                                                                           (int*) indices, (T_SCORE*) scores, (T_BBOX*) bboxData,
                                                                           /*(int*) keepCount,*/ (T_BBOX*) topDetections);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

// gatherTopDetectionsOpt LAUNCH CONFIG {{{
typedef ssdStatus_t (*gtdFunc)(cudaStream_t,
                               const bool,
                               const int,
                               const int,
                               const int,
                               const int,
                               const int,
                               const void*,
                               const void*,
                               const void*,
                               void*);
struct gtdLaunchConfig
{
    DType_t t_bbox;
    DType_t t_score;
    gtdFunc function;

    gtdLaunchConfig(DType_t t_bbox, DType_t t_score)
        : t_bbox(t_bbox)
        , t_score(t_score)
    {
    }
    gtdLaunchConfig(DType_t t_bbox, DType_t t_score, gtdFunc function)
        : t_bbox(t_bbox)
        , t_score(t_score)
        , function(function)
    {
    }
    bool operator==(const gtdLaunchConfig& other)
    {
        return t_bbox == other.t_bbox && t_score == other.t_score;
    }
};

using nvinfer1::DataType;

static std::vector<gtdLaunchConfig> gtdFuncVec;

bool gtdOptInit()
{
    gtdFuncVec.push_back(gtdLaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                         gatherTopDetectionsOpt_gpu<float, float>));
    return true;
}

static bool initialized = gtdOptInit();

//}}}

ssdStatus_t gatherTopDetectionsOpt(
    cudaStream_t stream,
    const bool shareLocation,
    const int numImages,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const DType_t DT_BBOX,
    const DType_t DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* topDetections)
{
    gtdLaunchConfig lc = gtdLaunchConfig(DT_BBOX, DT_SCORE);
    for (unsigned i = 0; i < gtdFuncVec.size(); ++i)
    {
        if (lc == gtdFuncVec[i])
        {
            DEBUG_PRINTF("gatherTopDetectionsOpt kernel %d\n", i);
            return gtdFuncVec[i].function(stream,
                                          shareLocation,
                                          numImages,
                                          numPredsPerClass,
                                          numClasses,
                                          topK,
                                          keepTopK,
                                          indices,
                                          scores,
                                          bboxData,
                                          //keepCount,
                                          topDetections);
        }
    }
    return STATUS_BAD_PARAM;
}

} // namespace plugin
} // namespace nvinfer1
