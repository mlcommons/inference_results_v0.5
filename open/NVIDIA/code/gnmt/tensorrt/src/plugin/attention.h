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

#ifndef TRT_ATTENTION_H
#define TRT_ATTENTION_H
#include <NvInfer.h>
#include <array>
#include <vector>

//!
//! \brief This structure is used to hold the parameters of the layer that are known at build time.
//!
struct AttentionParams
{
    int nbElements;
    std::array<std::vector<int>, 4> inputStrides;
    std::vector<int> resStrides;
    std::vector<int> len;
    nvinfer1::DataType type;
};

//!
//! \brief This structure holds the stride size of each of the inputs and the output in one dimension. An array of this
//!        structure can be used to represent the strides for the entire computation.
//!
struct Strides
{
    int input1;
    int input2;
    int input3;
    int input4;
    int res;
}; 

//!
//! \brief This structure holds the kernel parameters that are accessed from the device code. The members of this
//! structure
//!        are all fixed sized to increase ease of access from the device.
//!
struct KernelParams
{
    int nbElements;          //! Represents the total number of elements in the output
    int nbDims;              //! Number of dimensions
    Strides stridesArray[8]; //! Strides in the input tensors
    int len[8];              //! Length of each dimension
};

void launchAttentionFusedKernel(AttentionParams params, int batchSize, const void* const* inputs, void* output, cudaStream_t stream);
#endif
