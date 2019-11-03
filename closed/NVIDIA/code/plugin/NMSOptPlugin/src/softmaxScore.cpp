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

#include <cudnn.h>

#include "ssdOpt.h"
#include "ssdOptMacros.h"
namespace nvinfer1
{
namespace plugin
{

ssdStatus_t softmaxScore(cudaStream_t stream,
                        const int N,
                        const int num_classes,
                        const int num_priors,
                        const int num_dims,
                        const DType_t DT_DATA,
                        const void * in_scores,
                        void * out_scores,
                        cudnnHandle_t handle,
                        cudnnTensorDescriptor_t inScoreDesc,
                        cudnnTensorDescriptor_t outScoreDesc
                    
 )
{

    ASSERT(DT_DATA == DataType::kFLOAT);

    cudnnStatus_t status;
    status = cudnnSetStream(handle, stream);
    assert(status == CUDNN_STATUS_SUCCESS);
    status = cudnnSetTensor4dDescriptor(inScoreDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, num_classes, num_priors, num_dims);
    assert(status == CUDNN_STATUS_SUCCESS);
    status = cudnnSetTensor4dDescriptor(outScoreDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, num_classes, num_priors, num_dims);
    assert(status == CUDNN_STATUS_SUCCESS);

    cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL;
    float alpha = 1.0, beta = 0.0;
    status = cudnnSoftmaxForward(
        handle,
        algorithm,
        mode,
        &alpha,
        inScoreDesc,
        in_scores,
        &beta,
        outScoreDesc,
        out_scores);

    assert(status == CUDNN_STATUS_SUCCESS);
    return STATUS_SUCCESS; 
}

}

}
