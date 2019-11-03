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

#include "cuda_fp16.h"
#include "attention.h"
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

__device__ void warpReduce(float& val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
}

template <typename t_data, int NEXT_H, bool EXACT_H, int NUM_THREADS>
__global__ void attention_with_gemv(int MAX_S, int h, int beam_width, const t_data* encoder_input, const t_data* decoder_input, const t_data* bias, const t_data* vec, t_data* output) {
    int sample = blockIdx.y;
    int tok = blockIdx.x;

    __shared__ float sh_reduced[NUM_THREADS/32];

    float r_bias[NEXT_H/NUM_THREADS];
    float r_vec[NEXT_H/NUM_THREADS];
    float r_enc[NEXT_H/NUM_THREADS];
    
    const int TEMPLATED_STEP_SIEE = 256;

#pragma unroll
    for (int i=0;i<NEXT_H/NUM_THREADS;i++) {
        int hi=i*NUM_THREADS+threadIdx.x;
        r_bias[i] = bias[hi];
        r_vec[i] = vec[hi];
        if (EXACT_H || i < (NEXT_H - TEMPLATED_STEP_SIEE) / NUM_THREADS || hi < h) {            
            r_enc[i] = encoder_input[sample*MAX_S*h + tok*h + hi];
        }
    }

    for (int bm=0; bm<beam_width; bm++) {
        float r_dec[NEXT_H/NUM_THREADS];
#pragma unroll
        for (int i=0; i<NEXT_H/NUM_THREADS; i++) {
            int hi=i*NUM_THREADS+threadIdx.x;
            if (EXACT_H || i < (NEXT_H - TEMPLATED_STEP_SIEE) / NUM_THREADS || hi < h) {
                r_dec[i] = decoder_input[sample*beam_width*h + bm*h + hi];
            }
        }
        float partial_accum = 0.f;
#pragma unroll
        for (int i=0; i<NEXT_H/NUM_THREADS; i++) {
            int hi=i*NUM_THREADS+threadIdx.x;
            if (EXACT_H || i < (NEXT_H - TEMPLATED_STEP_SIEE) / NUM_THREADS || hi < h) {
                partial_accum += tanh(r_enc[i] + r_dec[i] + r_bias[i]) * r_vec[i];
            }
        }
        warpReduce(partial_accum);
        if ((threadIdx.x%32)==0) sh_reduced[threadIdx.x/32] = partial_accum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float accum = 0.f;
#pragma unroll
            for (int i=0; i<NUM_THREADS/32; i++) {
                accum += sh_reduced[i];
            }
            output[sample*MAX_S*beam_width+bm*MAX_S+tok] = accum;
        }
        __syncthreads();
    }
}


void launchAttentionFusedKernel(AttentionParams params, int batchSize, const void* const* inputs, void* output, cudaStream_t stream)
{
    //! Copy the attention parameters to the kernel parameters
    KernelParams p;
    p.nbElements = params.nbElements * batchSize;
    p.nbDims = params.inputStrides[0].size();
    for (int i = 0; i < p.nbDims; i++)
    {
        p.stridesArray[i].input1 = params.inputStrides[0][i];
        p.stridesArray[i].input2 = params.inputStrides[1][i];
        p.stridesArray[i].input3 = params.inputStrides[2][i];
        p.stridesArray[i].input4 = params.inputStrides[3][i];
        p.stridesArray[i].res = params.resStrides[i];
        p.len[i] = params.len[i];
    }

    int beam_width = params.len[1];
    int MAX_S = params.len[2];
    
    const int BLOCK_SIZE = 128;
    
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(MAX_S, batchSize);
    
    int h = params.inputStrides[0][2];
    
    nvinfer1::DataType eltype = params.type;
    
    if (eltype == nvinfer1::DataType::kFLOAT) {
        if (h % 256 == 0) {       
            switch (h) {
                case 256: 
                    attention_with_gemv<float, 256, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output); 
                    break;
                case 512: 
                    attention_with_gemv<float, 512, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
                    break;
                case 768: 
                    attention_with_gemv<float, 768, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output); 
                    break;
                case 1024: 
                    attention_with_gemv<float, 1024, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
                    break;
                case 1280: 
                    attention_with_gemv<float, 1280, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
                    break;
                case 1536: 
                    attention_with_gemv<float, 1536, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output); 
                    break;
                case 1792: 
                    attention_with_gemv<float, 1792, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
                    break;
                case 2048: 
                    attention_with_gemv<float, 2048, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
                    break;
                default:
                    assert(false);        
            }
        }
        else {
            if (h < 256) {
                attention_with_gemv<float, 256, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 512) {
                attention_with_gemv<float, 512, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 768) {
                attention_with_gemv<float, 768, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 1024) {
                attention_with_gemv<float, 1024, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 1280) {
                attention_with_gemv<float, 1280, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 1536) {
                attention_with_gemv<float, 1536, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 1792) {
                attention_with_gemv<float, 1792, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else if (h < 2048) {
                attention_with_gemv<float, 2048, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (const float*)inputs[3], (float*)output);
            }
            else {
                 assert(false);
            }
        }
    }
    else if (eltype == nvinfer1::DataType::kHALF) {
        if (h % 256 == 0) {       
            switch (h) {
                case 256: 
                    attention_with_gemv<half, 256, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output); 
                    break;
                case 512: 
                    attention_with_gemv<half, 512, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
                    break;
                case 768: 
                    attention_with_gemv<half, 768, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output); 
                    break;
                case 1024: 
                    attention_with_gemv<half, 1024, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
                    break;
                case 1280: 
                    attention_with_gemv<half, 1280, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
                    break;
                case 1536: 
                    attention_with_gemv<half, 1536, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output); 
                    break;
                case 1792: 
                    attention_with_gemv<half, 1792, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
                    break;
                case 2048: 
                    attention_with_gemv<half, 2048, true, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
                    break;
                default:
                    assert(false);        
            }
        }
        else {
            if (h < 256) {
                attention_with_gemv<half, 256, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 512) {
                attention_with_gemv<half, 512, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 768) {
                attention_with_gemv<half, 768, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 1024) {
                attention_with_gemv<half, 1024, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 1280) {
                attention_with_gemv<half, 1280, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 1536) {
                attention_with_gemv<half, 1536, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 1792) {
                attention_with_gemv<half, 1792, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else if (h < 2048) {
                attention_with_gemv<half, 2048, false, BLOCK_SIZE> <<< gridSize, blockSize, 0, stream >>> (MAX_S, h, beam_width, (const half*)inputs[0], (const half*)inputs[1], (const half*)inputs[2], (const half*)inputs[3], (half*)output);
            }
            else {
                 assert(false);
            }
        }
    }

    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
