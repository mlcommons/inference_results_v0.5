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

#ifndef GNMT_SCORER_PLUGIN_KERNEL_H
#define GNMT_SCORER_PLUGIN_KERNEL_H

#include <cuda.h>

// Reduces per-ray top-k into per-sample top-k taking into account length penalty
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
    int * candidateIndices);

#endif
