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

#include "multiGatherPluginKernel.h"
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <cuda.h>
#include <numeric>

template<int TENSOR_COUNT>
struct TensorParams
{
    int vectorLengths[TENSOR_COUNT];
    int srcVectorCounts[TENSOR_COUNT];
    const void* src[TENSOR_COUNT];
    void* dst[TENSOR_COUNT];
};

template<typename DATATYPE, int TENSOR_COUNT, int THREADBLOCK_SIZE>
__global__ void multi_gather(
    int batchSize,
    int outputVectorCount,
    int maxVectorLength,
    TensorParams<TENSOR_COUNT> tensorParams,
    const int * __restrict__ indices
)
{
    int sampleId = blockIdx.z;
    int tensorId = blockIdx.y;
    int outputVectorAndElemId = blockIdx.x * THREADBLOCK_SIZE + threadIdx.x;
    int outputVectorId = outputVectorAndElemId / maxVectorLength;
    int elemId = outputVectorAndElemId - outputVectorId * maxVectorLength;
    if ((outputVectorId < outputVectorCount) && (elemId < tensorParams.vectorLengths[tensorId]))
    {
        int inputVectorId = *(indices + (sampleId * outputVectorCount + outputVectorId));
        const DATATYPE * src_ptr = (const DATATYPE*)(tensorParams.src[tensorId]) + ((sampleId * tensorParams.srcVectorCounts[tensorId] + inputVectorId) * tensorParams.vectorLengths[tensorId] + elemId);
        DATATYPE * dst_ptr = (DATATYPE*)(tensorParams.dst[tensorId]) + ((sampleId * outputVectorCount + outputVectorId) * tensorParams.vectorLengths[tensorId] + elemId);
        *dst_ptr = *(src_ptr);
    }
}

int runMultiGather(
    cudaStream_t stream,
    int batchSize,
    int tensorCount,
    const int * vectorLengths,
    const int * srcVectorCounts,
    int outputVectorCount,
    int elemSize,
    const void * const * src,
    void ** dst,
    const int * indices)
{
    const int THREADBLOCK_SIZE = 256;
    const int TENSOR_COUNT = 16;

    // Single kernel run is capable of processing up to TENSOR_COUNT tensors, we might need multiple runs
    for(int baseTensorId = 0; baseTensorId < tensorCount; baseTensorId += TENSOR_COUNT)
    {
        TensorParams<TENSOR_COUNT> tensorParams;

        int currentTensorCount = std::min(TENSOR_COUNT, tensorCount - baseTensorId);
        for(int i = 0; i < currentTensorCount; ++i)
        {
            tensorParams.vectorLengths[i] = vectorLengths[baseTensorId + i];
            tensorParams.srcVectorCounts[i] = srcVectorCounts[baseTensorId + i];
            tensorParams.src[i] = src[baseTensorId + i];
            tensorParams.dst[i] = dst[baseTensorId + i];
        }

        int currentElemSize = elemSize;

        // Adjust element size and vector length to allow each thread of the kernel to transfer as large chunk of data as possible
        int nonZeroes = std::accumulate(tensorParams.vectorLengths, tensorParams.vectorLengths + currentTensorCount, 0, [] (int x, int y) {return x|y;});
        while (((nonZeroes & 1) == 0) && (currentElemSize < 16))
        {
            nonZeroes >>= 1;
            currentElemSize <<= 1;
        }
        for(int i = 0; i < currentTensorCount; ++i)
            tensorParams.vectorLengths[i] /= (currentElemSize / elemSize);

        int maxVectorLength = *std::max_element(tensorParams.vectorLengths, tensorParams.vectorLengths + currentTensorCount);
        dim3 gridSize((maxVectorLength * outputVectorCount + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE, currentTensorCount, batchSize);

        switch (currentElemSize)
        {
            case 2:
                multi_gather<unsigned short,TENSOR_COUNT,THREADBLOCK_SIZE><<<gridSize,THREADBLOCK_SIZE,0,stream>>>(
                    batchSize,
                    outputVectorCount,
                    maxVectorLength,
                    tensorParams,
                    indices);
                break;
            case 4:
                multi_gather<unsigned int,TENSOR_COUNT,THREADBLOCK_SIZE><<<gridSize,THREADBLOCK_SIZE,0,stream>>>(
                    batchSize,
                    outputVectorCount,
                    maxVectorLength,
                    tensorParams,
                    indices);
                break;
            case 8:
                multi_gather<unsigned long long,TENSOR_COUNT,THREADBLOCK_SIZE><<<gridSize,THREADBLOCK_SIZE,0,stream>>>(
                    batchSize,
                    outputVectorCount,
                    maxVectorLength,
                    tensorParams,
                    indices);
                break;
            case 16:
                multi_gather<ulonglong2,TENSOR_COUNT,THREADBLOCK_SIZE><<<gridSize,THREADBLOCK_SIZE,0,stream>>>(
                    batchSize,
                    outputVectorCount,
                    maxVectorLength,
                    tensorParams,
                    indices);
                break;
            default:
                assert(0);
                break;
        }
    }
            
    return 0;
}
