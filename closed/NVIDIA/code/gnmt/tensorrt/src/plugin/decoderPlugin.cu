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

#include <cuda.h>

#include "decoderPlugin.h"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)




using namespace nvinfer1;
using nvinfer1::plugin::GNMTDecoderPlugin;
using nvinfer1::plugin::GNMTDecoderPluginCreator;

REGISTER_TENSORRT_PLUGIN(GNMTDecoderPluginCreator);

GNMTDecoderPlugin::GNMTDecoderPlugin(const PluginFieldCollection *fc) {
    int idx = 0;
    
    mNumLayers = *(int*)(fc->fields[idx].data);
    idx++;
    
    mHiddenSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mAttentionSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mBeamSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mDataType = *(nvinfer1::DataType*)(fc->fields[idx].data);
    idx++;
    
    mWeights_h = (void**)malloc(mNumLayers * sizeof(void*));
    
    for (int i = 0; i < mNumLayers; i++) {        
        mWeights_h[i] = (void*)fc->fields[idx].data;
        idx++;
    }
    
    if (mDataType == DataType::kINT8) {       
        mPostActivationScalesH_h = (float*)(fc->fields[idx].data);
        idx++;                             
                                           
        mPostActivationScalesY_h = (float*)(fc->fields[idx].data);
        idx++;
        
        mLayerGemm0ScaleInput = *(float*)(fc->fields[idx].data);
        idx++;
        
        mLayerGemm0ScaleAttn = *(float*)(fc->fields[idx].data);
        idx++;
    }
    
}

GNMTDecoderPlugin::GNMTDecoderPlugin(const void* data, size_t length) {
    const char *d = static_cast<const char*>(data), *a = d;
    read<int>(d, mNumLayers);
    read<int>(d, mHiddenSize);
    read<int>(d, mAttentionSize);
    read<int>(d, mInputSize);
    read<int>(d, mBeamSize);
    
    read<nvinfer1::DataType>(d, mDataType);
    
    mPostActivationScalesH_h = (float*)malloc(mNumLayers * sizeof(float));
    mPostActivationScalesY_h = (float*)malloc(mNumLayers * sizeof(float));
    
    mWeights_h = (void**)malloc(mNumLayers * sizeof(void*));
    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
        else if (mDataType == DataType::kINT8) {
            dataTypeSize = sizeof(int8_t);
        }
        
        size_t sz = 4 * mHiddenSize * (mAttentionSize + 2 * mHiddenSize) * dataTypeSize;

        mWeights_h[i] = malloc(sz);
        memcpy(mWeights_h[i], d, sz);
        d += sz;
    }
    
    if (mDataType == DataType::kINT8) {       
        size_t sz = mNumLayers * sizeof(float);
        
        memcpy(mPostActivationScalesH_h, d, sz);
        d += sz;
        
        memcpy(mPostActivationScalesY_h, d, sz);
        d += sz;
        
        read<float>(d, mLayerGemm0ScaleInput);
        read<float>(d, mLayerGemm0ScaleAttn);
    }
    


    
    assert(d == a + length);  
}

const char* GNMTDecoderPlugin::getPluginType() const {
    return "GNMTDecoderPlugin";
}

const char* GNMTDecoderPlugin::getPluginVersion() const {
    return "1";
}

void GNMTDecoderPlugin::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* GNMTDecoderPlugin::getPluginNamespace() const {
    return mNamespace.c_str();
}

void GNMTDecoderPlugin::destroy() {
    if (mWeights_h) {
        free(mWeights_h);
        mWeights_h = nullptr;
    }
    if (mPostActivationScalesH_h) {
        free(mPostActivationScalesH_h);
        mPostActivationScalesH_h = nullptr;
    }
    if (mPostActivationScalesY_h) {
        free(mPostActivationScalesY_h);
        mPostActivationScalesY_h = nullptr;
    }

    delete this;
}

void GNMTDecoderPlugin::setCUDAInfo(cudaStream_t mStreami, cudaStream_t mStreamh, cudaStream_t* mSplitKStreams, cudaEvent_t* mSplitKEvents, cublasHandle_t mCublas, cublasLtHandle_t mCublasLt, void **mWeights_d, float *mPostActivationScalesH_d, float* mPostActivationScalesY_d) {
    this->mStreami = mStreami;
    this->mStreamh = mStreamh;
    this->mSplitKStreams = mSplitKStreams;
    this->mSplitKEvents = mSplitKEvents;
    this->mCublas = mCublas;
    this->mCublasLt = mCublasLt;
    this->mWeights_d = mWeights_d;
    this->mPostActivationScalesH_d = mPostActivationScalesH_d;
    this->mPostActivationScalesY_d = mPostActivationScalesY_d;
}

IPluginV2IOExt* GNMTDecoderPlugin::clone() const {
    size_t sz = getSerializationSize();
    
    char *buff = (char*)malloc(getSerializationSize());
    
    serialize(buff);
   
    GNMTDecoderPlugin* ret = new GNMTDecoderPlugin(buff, sz);
    
    ret->setCUDAInfo(mStreami, mStreamh, mSplitKStreams, mSplitKEvents, mCublas, mCublasLt, mWeights_d, mPostActivationScalesH_d, mPostActivationScalesY_d);
    
    free(buff);
    
    return ret;
}

int GNMTDecoderPlugin::getNbOutputs() const {
    return 1 + 2 * mNumLayers;
}

// TODO: No idea if this needs batch size. Actually, don't know what's expected at all.
Dims GNMTDecoderPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(index >= 0 && index < this->getNbOutputs());
    
    // y/hy/cy are all hiddenSize * batch.
    return Dims3(inputs[0].d[0], 1, mHiddenSize);
}


bool GNMTDecoderPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const { 
    if (inOut[pos].format != TensorFormat::kNCHW)
        return false;

    // fp16 I/O
    if (mDataType == nvinfer1::DataType::kHALF) {
        bool allHalf = true;

        // Don't care about pos. If all are half pass it.
        // The way this is called doesn't fill all of inOut, it only fills it up to pos.
        for (int i = 0; i <= pos; i++) {
            if (inOut[i].type != DataType::kHALF) {
                allHalf = false;
            }
        }
        
        if (allHalf) {
            return true;
        }
        return false;
    }
    else if (mDataType == nvinfer1::DataType::kINT8) {
        int localPos = pos;


        // Inputs
        // x
        if      (localPos == 0 && inOut[pos].type != DataType::kHALF) return false;
        // concatData
        else if (localPos == 1 && inOut[pos].type != DataType::kHALF) return false;
        // hx
        else if (localPos >= 2 && localPos < 2 + mNumLayers && inOut[pos].type != DataType::kHALF) return false;
        // cx
        else if (localPos >= 2 + mNumLayers && localPos < 2 + 2 * mNumLayers && inOut[pos].type != DataType::kHALF) return false;
        // bias
        else if (localPos >= 2 + 2 * mNumLayers && localPos < 2 + 3 * mNumLayers && inOut[pos].type != DataType::kFLOAT) return false;
        // preActivationScale
        else if (localPos >= 2 + 3 * mNumLayers && localPos < 2 + 4 * mNumLayers && inOut[pos].type != DataType::kFLOAT) return false;
        
        
        localPos -= nbInputs;
        
        // Outputs
        // y
        if      (localPos == 0 && inOut[pos].type != DataType::kHALF) return false;
        // hy
        else if (localPos >= 1 && localPos < 1 + mNumLayers && inOut[pos].type != DataType::kHALF) return false;
        // cy
        else if (localPos >= 1 + mNumLayers && localPos < 1 + 2 * mNumLayers && inOut[pos].type != DataType::kHALF) return false;
        
        return true;
    }
    return false;
}

void GNMTDecoderPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) {
    mInputSize = in[0].dims.d[in[0].dims.nbDims - 1];
}

void GNMTDecoderPlugin::configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast, const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
    mInputSize = inputDims[0].d[inputDims[0].nbDims - 1];
}



int GNMTDecoderPlugin::initialize() {
    CHECK(cublasCreate(&mCublas));
    CHECK(cublasLtCreate(&mCublasLt));
    
    CHECK(cublasSetMathMode(mCublas, CUBLAS_TENSOR_OP_MATH));
    
    CHECK(cudaStreamCreateWithPriority(&mStreami, 0, -1));
    CHECK(cudaStreamCreate(&mStreamh));
    mSplitKStreams = (cudaStream_t*)malloc(NUM_SPLIT_K_STREAMS * sizeof(cudaStream_t));
    mSplitKEvents = (cudaEvent_t*)malloc(NUM_SPLIT_K_STREAMS * sizeof(cudaEvent_t));

    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++)
    {
        CHECK(cudaStreamCreateWithPriority(&mSplitKStreams[i], 0, -1));
    }
    
        
    
    mWeights_d = (void**)malloc(mNumLayers * sizeof(void*));
    
    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
        else if (mDataType == DataType::kINT8) {
            dataTypeSize = sizeof(int8_t);
        }
        
        size_t sz = 4 * mHiddenSize * (mAttentionSize + 2 * mHiddenSize) * dataTypeSize;
        CHECK(cudaMalloc(&mWeights_d[i], sz));
        
        if (mDataType == DataType::kINT8) {           
            int8_t *tmpWeights;
            CHECK(cudaMalloc(&tmpWeights, sz));
            CHECK(cudaMemcpy(tmpWeights, mWeights_h[i], sz, cudaMemcpyHostToDevice));        
                              
                    
            // Layer
            {
                for (int splitK = 0; splitK < 2; splitK++) {               
                    int offset = splitK * 4 * mHiddenSize * mHiddenSize * sizeof(int8_t);
                    
                    int n = 4 * mHiddenSize;
                    int k = mHiddenSize;
                    int ldb = n;
                    
                    cublasLtMatrixLayout_t Bdesc = NULL;
                  
                    cublasLtMatrixTransformDesc_t transformDesc = NULL;
                    cublasLtMatrixLayout_t BtransformDesc = NULL;
                    float transformBAlpha = 1.0f;
                    float transformBBeta = 0.0f;
                    
                    cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;
                    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
                    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

                  
                    int ldbtransform = 32 * n;
                  
                    cublasErrCheck(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));
                    cublasErrCheck(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, ldb));
                    cublasErrCheck(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
                    
                    cublasErrCheck(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder)));
                    cublasErrCheck(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
                    
                    cublasErrCheck(cublasLtMatrixTransform(mCublasLt, transformDesc, &transformBAlpha, tmpWeights + offset, Bdesc, &transformBBeta, NULL, NULL, (int8_t*)(mWeights_d[i]) + offset, BtransformDesc, 0));
                    
                    
                    CHECK(cublasLtMatrixTransformDescDestroy(transformDesc));
                    CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
                    CHECK(cublasLtMatrixLayoutDestroy(BtransformDesc));
                }
            }
            
            // Recurrent. Only thing different is the offset.
            {
                int offset = 4 * mHiddenSize * (2 * mHiddenSize) * sizeof(int8_t);
                
                int n = 4 * mHiddenSize;
                int k = mHiddenSize;
                int ldb = n;
                
                cublasLtMatrixLayout_t Bdesc = NULL;
                
                cublasLtMatrixTransformDesc_t transformDesc = NULL;
                cublasLtMatrixLayout_t BtransformDesc = NULL;
                float transformBAlpha = 1.0f;
                float transformBBeta = 0.0f;
                
                cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;
                cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
                cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

                
                int ldbtransform = 32 * n;
                
                cublasErrCheck(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));
                cublasErrCheck(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, ldb));
                cublasErrCheck(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
                
                cublasErrCheck(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder, sizeof(colOrder)));
                cublasErrCheck(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));
                
                cublasErrCheck(cublasLtMatrixTransform(mCublasLt, transformDesc, &transformBAlpha, tmpWeights + offset, Bdesc, &transformBBeta, NULL, NULL, (int8_t*)(mWeights_d[i]) + offset, BtransformDesc, 0));
                
                
                CHECK(cublasLtMatrixTransformDescDestroy(transformDesc));
                CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
                CHECK(cublasLtMatrixLayoutDestroy(BtransformDesc));
            }
            
            CHECK(cudaFree(tmpWeights));
        }
        else { 
            CHECK(cudaMemcpy(mWeights_d[i], mWeights_h[i], sz, cudaMemcpyHostToDevice));
        }
        
        
    }

    if (mDataType == DataType::kINT8) {           
        size_t sz = mNumLayers * sizeof(float);
        CHECK(cudaMalloc(&mPostActivationScalesH_d, sz));
        CHECK(cudaMalloc(&mPostActivationScalesY_d, sz));
        
        CHECK(cudaMemcpy(mPostActivationScalesH_d, mPostActivationScalesH_h, sz, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(mPostActivationScalesY_d, mPostActivationScalesY_h, sz, cudaMemcpyHostToDevice));
    }
    return cudaSuccess;
}

void GNMTDecoderPlugin::terminate() {
    if (mCublas) {            
        CHECK(cublasDestroy(mCublas));
        mCublas = nullptr;
    }
    
    if (mStreami) {            
        CHECK(cudaStreamDestroy(mStreami));
        mStreami = nullptr;
    }
    if (mStreamh) {
        CHECK(cudaStreamDestroy(mStreamh));
        mStreamh = nullptr;
    }
            
    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++) {
        if (mSplitKStreams[i]) {               
            CHECK(cudaStreamDestroy(mSplitKStreams[i]));
            mSplitKStreams[i] = nullptr;
        }
    }

    if (mSplitKStreams) {           
        free(mSplitKStreams);
        mSplitKStreams = nullptr;
    }
    if (mSplitKEvents) {            
        free(mSplitKEvents);
        mSplitKEvents = nullptr;
    }
    
    if (mWeights_d) {
        for (int i = 0; i < mNumLayers; i++) {           
            if (mWeights_d[i]) {                
                cudaFree(mWeights_d[i]);
                mWeights_d[i] = nullptr;
            }
        }
        free(mWeights_d);
        mWeights_d = nullptr;
    }
    
    if (mPostActivationScalesH_d) {
        cudaFree(mPostActivationScalesH_d);
        mPostActivationScalesH_d = nullptr;
    }
    if (mPostActivationScalesY_d) {
        cudaFree(mPostActivationScalesY_d);
        mPostActivationScalesY_d = nullptr;
    }
}

size_t GNMTDecoderPlugin::getWorkspaceSize(int maxBatchSize) const {
    size_t size = 0;
    
    if (mDataType == nvinfer1::DataType::kHALF) {
        // tmp_io
        size += mNumLayers * (mAttentionSize + mInputSize) * maxBatchSize * mBeamSize * sizeof(half);
        
        // tmp_i
        size += mHiddenSize * maxBatchSize * mBeamSize * 4 * NUM_SPLIT_K_STREAMS * sizeof(half);
        
        // tmp_h
        size += mNumLayers * mHiddenSize * maxBatchSize * mBeamSize * 4 * sizeof(half);
    }
    else if (mDataType == nvinfer1::DataType::kINT8) {
        int effectiveBatch = maxBatchSize * mBeamSize;
        int roundedBatch = roundoff(effectiveBatch, 32);

        // tmp_io
        size += mNumLayers * (mAttentionSize + mInputSize) * roundedBatch * sizeof(int8_t);
        
        // tmp_i
        size += mHiddenSize * roundedBatch * 4 * NUM_SPLIT_K_STREAMS * sizeof(int32_t);
        
        // tmp_h
        size += mNumLayers * mHiddenSize * roundedBatch * 4 * sizeof(int32_t);

        // tmp_resid
        size += mNumLayers * mHiddenSize * roundedBatch * sizeof(float);
        
        // tmp_x
        size += mInputSize * roundedBatch * sizeof(int8_t);
        
        // tmp_y
        size += mHiddenSize * roundedBatch * sizeof(int8_t);
        
        // tmp_attention
        size += mAttentionSize * roundedBatch * sizeof(int8_t);
        
        // tmp_attention2
        size += mAttentionSize * roundedBatch * sizeof(int8_t);
        
        // tmp_h_in/out
        size += 2 * mNumLayers * mHiddenSize * roundedBatch * sizeof(int8_t);
        
        // tmp_i2
        size += 2 * mHiddenSize * roundedBatch * sizeof(int8_t);
    }

    return size;
}

int GNMTDecoderPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
    int effectiveBatch = batchSize * mBeamSize;

    assert(mAttentionSize == mHiddenSize);
    assert(mInputSize == mHiddenSize);

    void *tmp_io = NULL;
    void *tmp_i = NULL; 
    void *tmp_h = NULL; 
    void *tmp_resid = NULL; 
    void *tmp_x = NULL; 
    void *tmp_y = NULL; 
    void *tmp_attention = NULL;    
    void *tmp_attention2 = NULL;    
    
    void **tmp_h_in = NULL;
    void **tmp_h_out = NULL;
        
    void *tmp_i2 = NULL; 
    
    
    
    if (mDataType == nvinfer1::DataType::kHALF) {
        tmp_io = workspace;
        tmp_i = (void*)((char*)(workspace) + mNumLayers * (mAttentionSize + mInputSize) * effectiveBatch * sizeof(half));
        tmp_h = (void*)((char*)(tmp_i) + mHiddenSize * effectiveBatch * 4 * NUM_SPLIT_K_STREAMS * sizeof(half));
    }
    else if (mDataType == nvinfer1::DataType::kINT8) {
        int roundedBatch = roundoff(effectiveBatch, 32);
        
        tmp_io = workspace;
        tmp_i = (void*)((char*)(workspace) + mNumLayers * (mAttentionSize + mInputSize) * roundedBatch * sizeof(int8_t));
        tmp_h = (void*)((char*)(tmp_i) + mHiddenSize * roundedBatch * 4 * NUM_SPLIT_K_STREAMS * sizeof(int32_t));
        tmp_resid = (void*)((char*)(tmp_h) + mNumLayers * mHiddenSize * roundedBatch * 4 * sizeof(int32_t));
        
        tmp_x = (void*)((char*)(tmp_resid) + mNumLayers * mHiddenSize * roundedBatch * sizeof(float));
        tmp_y = (void*)((char*)(tmp_x) + mInputSize * roundedBatch * sizeof(int8_t));
        tmp_attention = (void*)((char*)(tmp_y) + mHiddenSize * roundedBatch * sizeof(int8_t));
        tmp_attention2 = (void*)((char*)(tmp_attention) + mAttentionSize * roundedBatch * sizeof(int8_t));
        
        
        tmp_h_in = (void**)malloc(mNumLayers * sizeof(void*));
        tmp_h_out = (void**)malloc(mNumLayers * sizeof(void*));
        
        tmp_h_in[0] = (void*)((char*)(tmp_attention2) + mAttentionSize * roundedBatch * sizeof(int8_t));
        tmp_h_in[1] = (void*)((char*)(tmp_h_in[0]) + mHiddenSize * roundedBatch * sizeof(int8_t));
        tmp_h_in[2] = (void*)((char*)(tmp_h_in[1]) + mHiddenSize * roundedBatch * sizeof(int8_t));
        
        tmp_h_out[0] = (void*)((char*)(tmp_h_in[2]) + mHiddenSize * roundedBatch * sizeof(int8_t));
        tmp_h_out[1] = (void*)((char*)(tmp_h_out[0]) + mHiddenSize * roundedBatch * sizeof(int8_t));
        tmp_h_out[2] = (void*)((char*)(tmp_h_out[1]) + mHiddenSize * roundedBatch * sizeof(int8_t));
        
        tmp_i2 =  (void*)((char*)(tmp_h_out[2]) + mHiddenSize * roundedBatch * sizeof(int8_t));
    }
    
    if (mDataType == nvinfer1::DataType::kINT8) { 
        float scaleX = mLayerGemm0ScaleInput;       

        float scaleAttn = mLayerGemm0ScaleAttn;
                
        float scale_h1 = mPostActivationScalesH_h[0];
        float scale_h2 = mPostActivationScalesH_h[1];
        float scale_h3 = mPostActivationScalesH_h[2];
        
        // Quantise and transform to the shape required by the GEMM
        bulk5DecoderTransformAndQuantize((int8_t*)tmp_x, (half*)inputs[0], scaleX, 
                                         (int8_t*)tmp_attention2, (half*)inputs[1], scaleAttn, 
                                         (int8_t*)tmp_h_in[0], (half*)inputs[2], scale_h1, 
                                         (int8_t*)tmp_h_in[1], (half*)inputs[3], scale_h2, 
                                         (int8_t*)tmp_h_in[2], (half*)inputs[4], scale_h3, 
                                         mHiddenSize, effectiveBatch, stream);
    }
    
    cudaEvent_t event;
    CHECK(cudaEventCreate(&event, cudaEventDisableTiming));
    CHECK(cudaEventRecord(event, stream));  
    CHECK(cudaStreamWaitEvent(mStreami, event, 0));
    CHECK(cudaStreamWaitEvent(mStreamh, event, 0));
    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++) {
        CHECK(cudaStreamWaitEvent(mSplitKStreams[i], event, 0));
    }
    CHECK(cudaEventDestroy(event));

    cudaError_t status;

    int inputSize = mInputSize + mAttentionSize;
   
    if (mDataType == nvinfer1::DataType::kHALF) {
        decoderStep<half, CUDA_R_16F, half, CUDA_R_16F, half>
                (mHiddenSize, 
                 inputSize,
                 effectiveBatch, 
                 1,
                 mNumLayers,
                 this->mCublas,
                 this->mCublasLt,
                 (half*)inputs[0], // x 
                 (half**)(&(inputs[2])), // Array of hx, 
                 (half**)(&inputs[2 + mNumLayers]), // Array of cx, 
                 (half**)mWeights_d,
                 (half**)(&inputs[2 + 2 * mNumLayers]), // bias
                 (half*)outputs[0], // y, 
                 (half**)(&outputs[1]), // Array of hy, 
                 (half**)(&outputs[1 + mNumLayers]), // Array of cy,
                 (half*)inputs[1], // attention,
                 (half*)tmp_io,
                 (half*)tmp_i,
                 (half*)tmp_h,
                 NULL,
                 NULL,
                 NULL,
                 NULL,
                 NULL,
                 mStreami,
                 mSplitKStreams,
                 mSplitKEvents,
                 NUM_SPLIT_K_STREAMS,
                 mStreamh);
    }
    else if (mDataType == nvinfer1::DataType::kINT8) { 
        decoderStep<int8_t, CUDA_R_8I, int32_t, CUDA_R_32I, float>
                (mHiddenSize, 
                 inputSize,
                 effectiveBatch, 
                 1,
                 mNumLayers,
                 this->mCublas,
                 this->mCublasLt,
                 (int8_t*)tmp_x, // x 
                 (int8_t**)(tmp_h_in), // Array of hx, 
                 (half**)(&inputs[2 + mNumLayers]), // Array of cx, 
                 (int8_t**)mWeights_d,
                 (float**)(&inputs[2 + 2 * mNumLayers]), // bias
                 (half*)outputs[0], //(int8_t*)tmp_y, // y, 
                 (half**)(&outputs[1]), // Array of hy, 
                 (half**)(&outputs[1 + mNumLayers]), // Array of cy,
                 (int8_t*)tmp_attention2, // attention,
                 (int8_t*)tmp_io,
                 (int32_t*)tmp_i,
                 (int32_t*)tmp_h,
                 (float*)tmp_resid,
                 (float**)(&inputs[2 + 3 * mNumLayers]), // gemm output scale,
                 mPostActivationScalesH_d, // postActivationScaleH,
                 mPostActivationScalesY_d, // postActivationScaleY,
                 (int8_t*)tmp_i2,
                 mStreami,
                 mSplitKStreams,
                 mSplitKEvents,
                 NUM_SPLIT_K_STREAMS,
                 mStreamh);
    }
             
    cudaEvent_t eventEnd;
    
    // The final kernel is the elementwise kernel launched to stream i, so only need to wait for that one to finish.
    CHECK(cudaEventCreate(&eventEnd, cudaEventDisableTiming));
    CHECK(cudaEventRecord(eventEnd, mStreami));  
    CHECK(cudaStreamWaitEvent(stream, eventEnd, 0));
    CHECK(cudaEventDestroy(eventEnd));  
    
    if (mDataType == nvinfer1::DataType::kINT8) { 
        free(tmp_h_in);
        free(tmp_h_out);
    }

    
    return 0;
}

size_t GNMTDecoderPlugin::getSerializationSize() const {
    size_t sz = sizeof(mNumLayers) + sizeof(mHiddenSize) + sizeof(mAttentionSize) + sizeof(mInputSize) + sizeof(mBeamSize) + sizeof(mDataType);
    
    // Weights
    for (int i = 0; i < mNumLayers; i++) {
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
        else if (mDataType == DataType::kINT8) {
            dataTypeSize = sizeof(int8_t);
        }
        
        sz += 4 * mHiddenSize * (mAttentionSize + 2 * mHiddenSize) * dataTypeSize;
    }
    
    // Scales
    if (mDataType == DataType::kINT8) {       
        sz += mNumLayers * sizeof(float);
        sz += mNumLayers * sizeof(float);
        sz += sizeof(float);
        sz += sizeof(float);
    }
    
    
    return sz;
}

void GNMTDecoderPlugin::serialize(void* buffer) const {
    char *d = static_cast<char*>(buffer), *a = d;

    write<int>(d, mNumLayers);
    write<int>(d, mHiddenSize);        
    write<int>(d, mAttentionSize);
    write<int>(d, mInputSize);
    write<int>(d, mBeamSize);
    write<nvinfer1::DataType>(d, mDataType);
    
    
    for (int i = 0; i < mNumLayers; i++) {        
        size_t dataTypeSize = 0;
        if (mDataType == DataType::kHALF) {
            dataTypeSize = sizeof(half);
        }
        else if (mDataType == DataType::kINT8) {
            dataTypeSize = sizeof(int8_t);
        }
        
        size_t sz = 4 * mHiddenSize * (mAttentionSize + 2 * mHiddenSize) * dataTypeSize;

        memcpy(d, mWeights_h[i], sz);
        d += sz;
    }
    
    if (mDataType == DataType::kINT8) {       
        size_t sz = mNumLayers * sizeof(float);
        
        memcpy(d, mPostActivationScalesH_h, sz);
        d += sz;
        
        memcpy(d, mPostActivationScalesY_h, sz);
        d += sz;

        write<float>(d, mLayerGemm0ScaleInput);
        write<float>(d, mLayerGemm0ScaleAttn);
    }
    

    assert(d == a + getSerializationSize());
}

nvinfer1::DataType GNMTDecoderPlugin::getOutputDataType (int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    return mDataType == nvinfer1::DataType::kINT8 ? nvinfer1::DataType::kHALF : mDataType;
}

bool GNMTDecoderPlugin::isOutputBroadcastAcrossBatch (int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const {
    return false;
}

bool GNMTDecoderPlugin::canBroadcastInputAcrossBatch (int inputIndex) const {
    return inputIndex >= 2 * mNumLayers + 2;
}

template <typename T>
void GNMTDecoderPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void GNMTDecoderPlugin::read(const char*& buffer, T& val) const
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

const char* GNMTDecoderPluginCreator::getPluginName() const {
    return "GNMTDecoderPlugin";
}

const char* GNMTDecoderPluginCreator::getPluginVersion() const {
    return "1";
}

const PluginFieldCollection* GNMTDecoderPluginCreator::getFieldNames() {
    return nullptr;        
}

void GNMTDecoderPluginCreator::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* GNMTDecoderPluginCreator::getPluginNamespace() const {
    return mNamespace.c_str();
}

IPluginV2IOExt* GNMTDecoderPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    return new GNMTDecoderPlugin(fc);        
}

IPluginV2IOExt* GNMTDecoderPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
    return new GNMTDecoderPlugin(serialData, serialLength);        
}

