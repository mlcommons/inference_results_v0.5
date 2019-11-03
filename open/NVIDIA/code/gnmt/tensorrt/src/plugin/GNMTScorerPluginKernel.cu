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

#include "GNMTScorerPluginKernel.h"
#include <cassert>
#include <cub/cub.cuh>

struct __align__(8) Candidate
{
    float logprob;
    int index;
};

template<int ELEMS_PER_THREAD, int THREADBLOCK_SIZE, typename intermediate_logprobs_type>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void final_reduction_with_length_penalty(
    const int * __restrict intermediateIndices,
    const intermediate_logprobs_type * __restrict intermediateLogprobs,
    const float * __restrict parentLogprobs,
    const float * __restrict lengthPenalties,
    float * __restrict candidateLogprobs,
    int * __restrict candidateIndices,
    int topK,
    int beamSize,
    int elemsPerRay,
    int vocSize,
    int eosIndex)
{
    int sample_id = blockIdx.x;
    int thread_id = threadIdx.x;

    intermediateIndices += sample_id * (beamSize * elemsPerRay);
    intermediateLogprobs += sample_id * (beamSize * elemsPerRay);
    parentLogprobs += sample_id * beamSize;

    typedef cub::BlockRadixSort<float, THREADBLOCK_SIZE, ELEMS_PER_THREAD, Candidate> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    float scores[ELEMS_PER_THREAD];
    Candidate candidates[ELEMS_PER_THREAD];

    float finishedLengthPenalty = lengthPenalties[0];
    float unfinishedLengthPenalty = lengthPenalties[1];

    int elem_id = thread_id;
    for(int i = 0; i < ELEMS_PER_THREAD; ++i, elem_id += THREADBLOCK_SIZE)
    {
        scores[i] = -FLT_MAX;
        if (elem_id < (beamSize * elemsPerRay)) // Checking if elem_id is within bounds and references a valid element
        {
            int rayId = elem_id / elemsPerRay;

            float intermediateLogprob;
            if (sizeof(intermediate_logprobs_type) == 4) // Checking if we are dealing with fp32 or fp16 input elements
                intermediateLogprob = intermediateLogprobs[elem_id];
            else
                intermediateLogprob = __half2float(intermediateLogprobs[elem_id]);
    

            int intermediateIndex = intermediateIndices[elem_id];

            candidates[i].logprob = intermediateLogprob + parentLogprobs[rayId];
            candidates[i].index = rayId * vocSize + intermediateIndex; // Encoding both rayId and token Id (from intermediateIndices) into a single index
            scores[i] = candidates[i].logprob * ((intermediateIndex == eosIndex) ? finishedLengthPenalty : unfinishedLengthPenalty);
        }
    }

    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(scores, candidates);

    if (thread_id < topK)
    {
        candidateLogprobs[sample_id * topK + thread_id] = candidates[0].logprob;
        candidateIndices[sample_id * topK + thread_id] = candidates[0].index;
    }
}

int runFinalReductionWithLengthPenalty(
    cudaStream_t stream,
    int batchSize,
    int topK,
    int beamSize,
    int elemsPerRay,
    int vocSize,
    int eosIndex,
    const int * intermediateIndices,
    const void * intermediateLogprobs,
    bool fp16intermediateLogprobs,
    const float * parentLogprobs,
    const float * lengthPenalties,
    float * candidateLogprobs,
    int * candidateIndices)
{
    assert(topK < 64);
    int elems = beamSize * elemsPerRay;

    if (fp16intermediateLogprobs)
    {
        if (elems <= 64)
            final_reduction_with_length_penalty<1, 64, __half><<<batchSize,64,0,stream>>>(
                intermediateIndices,
                (const __half *)intermediateLogprobs,
                parentLogprobs,
                lengthPenalties,
                candidateLogprobs,
                candidateIndices,
                topK,
                beamSize,
                elemsPerRay,
                vocSize,
                eosIndex);
        else if (elems <= 128)
            final_reduction_with_length_penalty<1, 128, __half><<<batchSize,128,0,stream>>>(
                intermediateIndices,
                (const __half *)intermediateLogprobs,
                parentLogprobs,
                lengthPenalties,
                candidateLogprobs,
                candidateIndices,
                topK,
                beamSize,
                elemsPerRay,
                vocSize,
                eosIndex);
        else if (elems <= 256)
            final_reduction_with_length_penalty<1, 256, __half><<<batchSize,256,0,stream>>>(
                intermediateIndices,
                (const __half *)intermediateLogprobs,
                parentLogprobs,
                lengthPenalties,
                candidateLogprobs,
                candidateIndices,
                topK,
                beamSize,
                elemsPerRay,
                vocSize,
                eosIndex);
        else
            assert(0);
    }
    else
    {
        if (elems <= 64)
            final_reduction_with_length_penalty<1, 64, float><<<batchSize,64,0,stream>>>(
                intermediateIndices,
                (const float *)intermediateLogprobs,
                parentLogprobs,
                lengthPenalties,
                candidateLogprobs,
                candidateIndices,
                topK,
                beamSize,
                elemsPerRay,
                vocSize,
                eosIndex);
        else if (elems <= 128)
            final_reduction_with_length_penalty<1, 128, float><<<batchSize,128,0,stream>>>(
                intermediateIndices,
                (const float *)intermediateLogprobs,
                parentLogprobs,
                lengthPenalties,
                candidateLogprobs,
                candidateIndices,
                topK,
                beamSize,
                elemsPerRay,
                vocSize,
                eosIndex);
        else if (elems <= 256)
            final_reduction_with_length_penalty<1, 256, float><<<batchSize,256,0,stream>>>(
                intermediateIndices,
                (const float *)intermediateLogprobs,
                parentLogprobs,
                lengthPenalties,
                candidateLogprobs,
                candidateIndices,
                topK,
                beamSize,
                elemsPerRay,
                vocSize,
                eosIndex);
        else
            assert(0);
    }
                
    return 0;    
}
