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

 
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

__host__ __device__ int roundoff(int v, int d) {
   return (v + d - 1) / d * d;
}
 
// Use cublasLtMatmul to perform the tensor op Igemm with the memory 
// order transforms on all buffers.
//
// For better performance the data order transforms should be offline 
// as much as possible.
//
// Transa, transb assumed N; alpha, beta are host pointers; Tensor ops 
// allowed. Alpha assumed 1, beta assumed 0, and stream assumed 0.

cublasStatus_t LtIgemmTensor(cublasLtHandle_t ltHandle,
             int m,
             int n,
             int k,
             const int8_t *A,
             int lda,
             const int8_t *B,
             int ldb,
             int32_t *C,
             int ldc,
             int8_t *tmp_A,
             const bool needsATransformed,
             cudaStream_t stream) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
  
    // The tensor op igemm kernels require specialized memory order of 
    // data.
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t *Atransform = (int8_t*)tmp_A;
    int8_t *Btransform = (int8_t*)B;
    int32_t *Ctransform = (int32_t*)C;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAAlpha = 1.0f;
    float transformABeta = 0.0f;
    
    
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;

  
    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;
    
    cublasErrCheck(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32I));
    
    // Tensor op igemm kernels only support NT gemm
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose));
    
      
    cublasErrCheck(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
    cublasErrCheck(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));
    
  
    cublasErrCheck(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
    cublasErrCheck(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
    
  
    cublasErrCheck(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
    cublasErrCheck(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));
  
    if (needsATransformed) {
        cublasErrCheck(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));
        
        cublasErrCheck(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
        cublasErrCheck(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, k, m, lda));    
        
        cublasErrCheck(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder)));
        cublasErrCheck(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAAlpha, A, Adesc, &transformABeta, NULL, NULL, Atransform, AtransformDesc, stream));
        
        cublasErrCheck(cublasLtMatrixTransformDescDestroy(transformDesc));
    }
    else {
        Atransform = (int8_t*)A;
    }
    
    cublasErrCheck(cublasLtMatmul(ltHandle,
                            matmulDesc,
                            &alpha,
                            Atransform,
                            AtransformDesc,
                            Btransform,
                            BtransformDesc,
                            &beta,
                            Ctransform,
                            CtransformDesc,
                            Ctransform,
                            CtransformDesc,
                            NULL,
                            NULL,
                            0,
                            stream));

    if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
    if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
    if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
   
    return status;
}


// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
    return 1.f / (1.f + __expf(-in));  
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}


__global__ void decoderQuantize_ker(int8_t* out, float* in, float scale, int cnt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < cnt) {
        out[index] = max(-128, min(127, (int)(in[index] * scale + 0.5f)));
    }   
}

void decoderQuantize(int8_t* out, float* in, float scale, int cnt, cudaStream_t stream) {   
    const int numThreads = 256;
    
    decoderQuantize_ker <<< (cnt + numThreads - 1) / numThreads, numThreads, 0, stream >>> (out, in, scale, cnt);    
}




// COL->COL32 with transpose.
__global__ void bulk5DecoderTransformAndQuantize_ker(int8_t* __restrict__ out0, half* __restrict__ in0, float scale0,
                                                     int8_t* __restrict__ out1, half* __restrict__ in1, float scale1,
                                                     int8_t* __restrict__ out2, half* __restrict__ in2, float scale2,
                                                     int8_t* __restrict__ out3, half* __restrict__ in3, float scale3,
                                                     int8_t* __restrict__ out4, half* __restrict__ in4, float scale4, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= rows * cols) return;
    
    int inRow = index % rows;
    int inCol = index / rows;
    
       
    int outCol = inRow;
    int outRow = inCol;
    
    int outCols = rows;
    int outRows = cols;
    
    int outIdx = ((outCol / 32) * outRows + outRow) * 32 + outCol % 32;
    
    out0[outIdx] = max(-128, min(127, (int)((float)(in0[index]) * scale0 + 0.5f)));    
    out1[outIdx] = max(-128, min(127, (int)((float)(in1[index]) * scale1 + 0.5f)));    
    out2[outIdx] = max(-128, min(127, (int)((float)(in2[index]) * scale2 + 0.5f)));    
    out3[outIdx] = max(-128, min(127, (int)((float)(in3[index]) * scale3 + 0.5f)));    
    out4[outIdx] = max(-128, min(127, (int)((float)(in4[index]) * scale4 + 0.5f)));    
}

void bulk5DecoderTransformAndQuantize(int8_t* out0, half* in0, float scale0,
                                      int8_t* out1, half* in1, float scale1,
                                      int8_t* out2, half* in2, float scale2,
                                      int8_t* out3, half* in3, float scale3,
                                      int8_t* out4, half* in4, float scale4, int rows, int cols, cudaStream_t stream) {   
    const int numThreads = 256;
    
    bulk5DecoderTransformAndQuantize_ker <<< (rows * cols + numThreads - 1) / numThreads, numThreads, 0, stream >>> (out0, in0, scale0, 
                                                                                                                     out1, in1, scale1, 
                                                                                                                     out2, in2, scale2, 
                                                                                                                     out3, in3, scale3, 
                                                                                                                     out4, in4, scale4, rows, cols);    
}



__global__ void decoderTransformAndQuantize_ker(int8_t* out, half* in, float scale, int rows, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= rows * cols) return;
    
    int inRow = index % rows;
    int inCol = index / rows;
    
       
    int outCol = inRow;
    int outRow = inCol;
    
    int outCols = rows;
    int outRows = cols;
    
    int outIdx = ((outCol / 32) * outRows + outRow) * 32 + outCol % 32;
    
    out[outIdx] = max(-128, min(127, (int)((float)(in[index]) * scale + 0.5f)));    
}

void decoderTransformAndQuantize(int8_t* out, half* in, float scale, int rows, int cols, cudaStream_t stream) {   
    const int numThreads = 256;
    
    decoderTransformAndQuantize_ker <<< (rows * cols + numThreads - 1) / numThreads, numThreads, 0, stream >>> (out, in, scale, rows, cols);    
}


__global__ void decoderQuantize_ker(int8_t* out, half* in, float scale, int cnt) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (index < cnt) {
        half2 inh2 = *((half2*)(&in[index]));
        
        *((uchar2*)(&out[index])) = make_uchar2(max(-128, min(127, (int)((float)(inh2.x) * scale + 0.5f))),
                                                max(-128, min(127, (int)((float)(inh2.y) * scale + 0.5f))));
    }   
}

void decoderQuantize(int8_t* out, half* in, float scale, int cnt, cudaStream_t stream) {   
    const int numThreads = 256;
    
    decoderQuantize_ker <<< (cnt / 2 + numThreads - 1) / numThreads, numThreads, 0, stream >>> (out, in, scale, cnt);    
}


__global__ void decoderDeQuantize_ker(float* out, int8_t* in, float scale, int cnt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < cnt) {
        out[index] = in[index] * scale;
    }   
}

void decoderDeQuantize(float* out, int8_t* in, float scale, int cnt, cudaStream_t stream) {   
    const int numThreads = 256;
    
    decoderDeQuantize_ker <<< (cnt + numThreads - 1) / numThreads, numThreads, 0, stream >>> (out, in, scale, cnt);    
}

__global__ void decoderDeQuantize_ker(half* out, int8_t* in, float scale, int cnt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < cnt) {
        out[index] = (float)in[index] * scale;
    }   
}

void decoderDeQuantize(half* out, int8_t* in, float scale, int cnt, cudaStream_t stream) {   
    const int numThreads = 256;
    
    decoderDeQuantize_ker <<< (cnt + numThreads - 1) / numThreads, numThreads, 0, stream >>> (out, in, scale, cnt);    
}

#define NUM_SPLIT_K_STREAMS 2

// Fused forward kernel
template<typename T_GEMM_IN, typename T_GEMM_OUT, typename T_BIAS, int blockSize>
__global__ void elementWise_fp_IMMA(int hiddenSize, 
                               int outputSize_i, 
                               int batchSize,
                               int numSplitKStreams,
                               T_GEMM_OUT *tmp_h, 
                               T_GEMM_OUT *tmp_i, 
                               T_BIAS *tmp_i_resid, 
                               T_BIAS *tmp_i_resid_out,
                               T_BIAS *bias,
                               half *h_out,
                               T_GEMM_IN *i_out,
                               half *y,
                               bool finalLayer,
                               half *c_in,
                               half *c_out,
                               float *preActivationScale,
                               float *postActivationScaleH,
                               float *postActivationScaleY,
                               int layer) {
    // Takes in batch-major COL32 format.
    int activation = blockIdx.x * 32 + threadIdx.x;
    int example = blockIdx.y * blockDim.y + threadIdx.y;
        
    T_GEMM_OUT g[4];
    float activationIn[4];

    float in_gate;   
    float forget_gate;
    float in_gate2;
    float out_gate;  
   
    if (example < batchSize && activation < hiddenSize) {
        int roundedBatch = roundoff(batchSize, 32);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int index = activation % 32 + i * hiddenSize * batchSize + example * 32 + (activation / 32) * 32 * batchSize;
            
            activationIn[i] = (int32_t)tmp_h[index] * preActivationScale[8 * hiddenSize + i * hiddenSize + activation];
            
            // TODO: Should probably template NUM_SPLIT_K_STREAMS rather than using the define
            #pragma unroll
            for (int j = 0; j < NUM_SPLIT_K_STREAMS; j++) {
                activationIn[i] += (int32_t)tmp_i[index + j * hiddenSize * roundedBatch * 4] * preActivationScale[j * 4 * hiddenSize + i * hiddenSize + activation]; 
            }
            
        }
            
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            activationIn[i] += (float)bias[i * hiddenSize + activation];
        }
        
        in_gate      = sigmoidf(activationIn[0]);
        forget_gate  = sigmoidf(1 + activationIn[1]);
        in_gate2     = tanhf(activationIn[2]);
        out_gate     = sigmoidf(activationIn[3]);
        
        int basicIndex = example * hiddenSize + activation;
        
        float val = (forget_gate * (float)c_in[basicIndex]) + (in_gate * in_gate2);
        
        c_out[basicIndex] = val;
        
        val = out_gate * tanhf(val);

        T_GEMM_IN outH;
        T_GEMM_IN outY;
        
        h_out[basicIndex] = val;

        if (finalLayer) {
            if (tmp_i_resid) {
                y[basicIndex] = val + (float)tmp_i_resid[basicIndex];
            }
            else {
                y[basicIndex] = val;
            }
        }
        else {           
            if (tmp_i_resid) {
                outY = max(-128, min(127, (int)((val + (float)tmp_i_resid[basicIndex]) * postActivationScaleY[layer] + 0.5f)));
            }
            else {
                outY = max(-128, min(127, (int)(val * postActivationScaleY[layer] + 0.5f)));
            }
            
            i_out[basicIndex] = outY;
        }
        
        
        if (tmp_i_resid_out) {
            if (tmp_i_resid) {
                tmp_i_resid_out[basicIndex] = val + (float)tmp_i_resid[basicIndex];
            }
            else {                
                tmp_i_resid_out[basicIndex] = val;
            }
        }
    }
}
    
template<typename T_GEMM_IN, typename T_GEMM_OUT, typename T_BIAS, int blockSize>
__global__ void elementWise_fp(int hiddenSize, 
                               int outputSize_i, 
                               int batchSize,
                               int numSplitKStreams,
                               T_GEMM_OUT *tmp_h, 
                               T_GEMM_OUT *tmp_i, 
                               T_BIAS *tmp_i_resid, 
                               T_BIAS *tmp_i_resid_out,
                               T_BIAS *bias,
                               half *h_out,
                               T_GEMM_IN *i_out,
                               half *y,
                               bool finalLayer,
                               half *c_in,
                               half *c_out,
                               int layer) {
    int numElements = batchSize * hiddenSize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= numElements) return;
    
    // TODO: fast divmod.
    int example = index / hiddenSize;
    int gateIndex = (index % hiddenSize) + 4 * example * hiddenSize;    
    
    T_GEMM_OUT g[4];
    float activationIn[4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        activationIn[i] = tmp_h[i * hiddenSize + gateIndex];
        
        g[i] = 0;
                
        // TODO: Should probably template NUM_SPLIT_K_STREAMS rather than using the define
        #pragma unroll
        for (int j = 0; j < NUM_SPLIT_K_STREAMS; j++) {
            g[i] += (T_GEMM_OUT)tmp_i[i * hiddenSize + gateIndex + j * numElements * 4]; 
        }
        
        activationIn[i] += (float)g[i];
    }
    
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        activationIn[i] += (float)bias[i * hiddenSize + index % hiddenSize];
    }
    
    
    float in_gate      = sigmoidf(activationIn[0]);
    float forget_gate  = sigmoidf(1 + activationIn[1]);
    float in_gate2     = tanhf(activationIn[2]);
    float out_gate     = sigmoidf(activationIn[3]);
    
    
    float val = (forget_gate * (float)c_in[index]) + (in_gate * in_gate2);
    
    c_out[index] = val;
    
    val = out_gate * tanhf(val);

    T_GEMM_IN outH;
    T_GEMM_IN outY;
    
    outH = val;
    h_out[index] = outH;
    
    if (tmp_i_resid) {
        outY = val + (float)tmp_i_resid[index];
    }
    else {
        outY = outH;
    }
    i_out[index] = outY;
}


template<typename T_GEMM_IN, cudaDataType_t dataTypeIn, typename T_GEMM_OUT, cudaDataType_t dataTypeOut, typename T_BIAS>
void decoderStep(int hiddenSize, 
              int inputSize,
              int batchSize, 
              int seqLength, 
              int numLayers,
              cublasHandle_t cublasHandle,
              cublasLtHandle_t cublasLtHandle,
              T_GEMM_IN *x, 
              T_GEMM_IN **hx, 
              half **cx, 
              T_GEMM_IN **w, 
              T_BIAS **bias,
              half *y, 
              half **hy, 
              half **cy,
              T_GEMM_IN *concatData,
              T_GEMM_IN *tmp_io,
              T_GEMM_OUT *tmp_i,
              T_GEMM_OUT *tmp_h,
              T_BIAS *tmp_resid,
              float **preActivationScale,
              float *postActivationScaleH,
              float *postActivationScaleY,
              T_GEMM_IN* tmp_i2,
              cudaStream_t streami,
              cudaStream_t* splitKStreams,
              cudaEvent_t* splitKEvents,
              int numSplitKStreams,
              cudaStream_t streamh) {
    T_GEMM_OUT alphaR = 1.f;
    T_GEMM_OUT betaR  = 0.f;    
    
    T_GEMM_OUT alphaL = 1.f;
    T_GEMM_OUT betaL  = 0.f;       

    int numElements = hiddenSize * batchSize;
    
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    
    if (seqLength > 1) {
        printf("Seq length > 1 not supported in this code.\n");
        return;
    }
    
    for (int layer = 0; layer < numLayers; layer++) {
        cudaEvent_t event;
       
        
        T_GEMM_IN *layer_i_in = layer == 0 ? x : tmp_io + numElements * (layer - 1);
        T_GEMM_IN *layer_i_out = layer == numLayers - 1 ? (T_GEMM_IN*)y : tmp_io + numElements * layer;
       
        // Run these in parallel with each other
        for (int i = 0; i < numSplitKStreams; i++) {           
            cublasErrCheck(cublasSetStream(cublasHandle, splitKStreams[i]));
            cudaErrCheck(cudaEventCreate(&splitKEvents[i], cudaEventDisableTiming));
            
            T_GEMM_IN *inData;
            
            if (i < numSplitKStreams / 2) {
                inData = layer_i_in + 2 * i * hiddenSize / numSplitKStreams;
            }
            else {
                inData = concatData;
            }
            
            if (dataTypeIn == CUDA_R_8I) {
                int8_t *A;
                int8_t *B;
                int32_t *C;
                

                int m = batchSize;
                int n = 4 * hiddenSize;
                int k = inputSize / numSplitKStreams;
                
                int lda = k;
                int ldb = k;
                int ldc = m;
                
                int roundedM = roundoff(m, 32);
                int roundedN = roundoff(n, 32);
                
                A = (int8_t*)inData;
                B = (int8_t*)(w[layer] + i * 4 * hiddenSize * inputSize / numSplitKStreams);
                C = (int32_t*)(tmp_i + i * n * roundedM);

                LtIgemmTensor(cublasLtHandle,
                              m, n, k,
                              A, lda,
                              B, ldb,
                              C, ldc,
                              (int8_t*)(tmp_i2) + i * roundedM * k,
                              layer != 0 && i < numSplitKStreams / 2,
                              splitKStreams[i]);

            }
            else {               
                T_GEMM_IN *wData;
                if (transa == CUBLAS_OP_N) {
                    wData = w[layer] + i * 4 * hiddenSize * inputSize / numSplitKStreams;
                }
                else {
                    wData = w[layer] + i * inputSize / numSplitKStreams;
                }
                
                cublasErrCheck(cublasGemmEx(cublasHandle,
                                            transa, transb,
                                            4 * hiddenSize, batchSize, inputSize / numSplitKStreams,
                                            &alphaL,
                                            wData,
                                            dataTypeIn,
                                            transa == CUBLAS_OP_N ? 4 * hiddenSize : inputSize,
                                            inData,
                                            dataTypeIn,
                                            hiddenSize,
                                            &betaL,
                                            tmp_i + 4 * i * numElements,
                                            dataTypeOut,
                                            4 * hiddenSize,
                                            dataTypeOut,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                
            }

            cudaErrCheck(cudaEventRecord(splitKEvents[i], splitKStreams[i]));  
        }
        
        // Converge split-K streams into stream i
        for (int i = 0; i < numSplitKStreams; i++) {           
            cudaErrCheck(cudaStreamWaitEvent(streami, splitKEvents[i], 0));
            cudaErrCheck(cudaEventDestroy(splitKEvents[i]));  
        }
        
        
        // Run recurrent GEMM in stream H (still in parallel with GEMMs above)
        cublasErrCheck(cublasSetStream(cublasHandle, streamh));
        
        if (dataTypeIn == CUDA_R_8I) {
            int8_t *A;
            int8_t *B;
            int32_t *C;
            
            int m = batchSize;
            int n = 4 * hiddenSize;
            int k = hiddenSize;
            
            int lda = k;
            int ldb = k;
            int ldc = m;
            
            int roundedM = roundoff(m, 32);
            int roundedN = roundoff(n, 32);
            
            A = (int8_t*)hx[layer];
            B = (int8_t*)(&w[layer][4 * hiddenSize * inputSize]);
            C = (int32_t*)(tmp_h + layer * n * roundedM);

            LtIgemmTensor(cublasLtHandle,
                          m, n, k,
                          A, lda,
                          B, ldb,
                          C, ldc,
                          NULL,
                          false,
                          streamh);


        }
        else {    
            cublasErrCheck(cublasGemmEx(cublasHandle,
                                        transa, transb,
                                        4 * hiddenSize, batchSize, hiddenSize,
                                        &alphaR,
                                        &w[layer][4 * hiddenSize * inputSize], 
                                        dataTypeIn,
                                        transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                        hx[layer],
                                        dataTypeIn,
                                        hiddenSize,
                                        &betaR,
                                        tmp_h + 4 * layer * numElements, 
                                        dataTypeOut,
                                        4 * hiddenSize,
                                        dataTypeOut,
                                        CUBLAS_GEMM_DEFAULT_TENSOR_OP)); 
        }
        
        cudaErrCheck(cudaEventCreate(&event, cudaEventDisableTiming));
        cudaErrCheck(cudaEventRecord(event, streamh));  
        
        dim3 blockDim;
        dim3 gridDim;
        

        // Stream i has hit the the convergence point above. Wait on stream i for stream h to also hit it.
        cudaErrCheck(cudaStreamWaitEvent(streami, event, 0));
        
        T_BIAS* resid_in;
        T_BIAS* resid_out;
        
        if (std::is_same<T_GEMM_IN, int8_t>::value) {
            resid_in = layer > 0 ? &tmp_resid[(layer - 1) * hiddenSize * batchSize] : NULL;
            resid_out = &tmp_resid[layer * hiddenSize * batchSize];
        }
        else {
            resid_in = layer > 0 ? (T_BIAS*)layer_i_in : NULL;
            resid_out = NULL;
        }
        
        if (std::is_same<T_GEMM_IN, int8_t>::value) {
            blockDim.x = 32;
            blockDim.y = 8;
            
            gridDim.x = (hiddenSize + blockDim.x - 1) / blockDim.x;
            gridDim.y = (batchSize + blockDim.y - 1) / blockDim.y;
            
            int roundedBatch = roundoff(batchSize, 32);
            
            
            elementWise_fp_IMMA<T_GEMM_IN, T_GEMM_OUT, T_BIAS, 256> <<< gridDim, blockDim , 0, streami >>> 
                     (hiddenSize, 
                      inputSize,
                      batchSize,
                      numSplitKStreams,
                      tmp_h + 4 * layer * hiddenSize * roundedBatch, 
                      tmp_i,
                      resid_in,
                      resid_out,
                      bias[layer],
                      hy[layer],
                      layer_i_out,
                      y,
                      layer == numLayers - 1,
                      cx[layer],
                      cy[layer],
                      preActivationScale == NULL ? NULL : preActivationScale[layer],
                      postActivationScaleH,
                      postActivationScaleY,
                      layer);
        }
        else {
            blockDim.x = 256;
            gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
            
            elementWise_fp<T_GEMM_IN, T_GEMM_OUT, T_BIAS, 256> <<< gridDim, blockDim , 0, streami >>> 
                     (hiddenSize, 
                      inputSize,
                      batchSize,
                      numSplitKStreams,
                      tmp_h + 4 * layer * numElements, 
                      tmp_i,
                      resid_in,
                      resid_out,
                      bias[layer],
                      hy[layer],
                      layer_i_out,
                      y,
                      layer == numLayers - 1,
                      cx[layer],
                      cy[layer],
                      layer);
        }
        cudaErrCheck(cudaGetLastError());
        
        // Split-k streams need to wait for eltwise op to complete 
        cudaErrCheck(cudaEventRecord(event, streami));
        for (int i = 0; i < numSplitKStreams; i++) {           
            cudaErrCheck(cudaStreamWaitEvent(splitKStreams[i], event, 0));
        }
        
        cudaErrCheck(cudaEventDestroy(event));  
    }
}


