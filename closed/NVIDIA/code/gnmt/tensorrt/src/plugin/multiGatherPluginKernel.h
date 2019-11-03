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

#ifndef MULTI_GATHER_PLUGIN_KERNEL_H
#define MULTI_GATHER_PLUGIN_KERNEL_H

#include <cuda.h>
#include <vector>

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
    const int * indices);

#endif
