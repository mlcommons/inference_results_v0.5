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

#ifndef GNMT_DECODER_PLUGIN_H
#define GNMT_DECODER_PLUGIN_H

#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <cublasLt.h>


#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <assert.h>

#include <iostream>
#include <string>

#include "decoderPlugin.cuh"

namespace nvinfer1
{

namespace plugin
{

class GNMTDecoderPlugin : public IPluginV2IOExt {
public:
    GNMTDecoderPlugin(const PluginFieldCollection *fc);

    // create the plugin at runtime from a byte stream
    GNMTDecoderPlugin(const void* data, size_t length);
    ~GNMTDecoderPlugin() override = default;

    const char *getPluginType() const override;
    
    const char *getPluginVersion() const override;
    
    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;
    
    void destroy() override;
    
    void setCUDAInfo(cudaStream_t mStreami, cudaStream_t mStreamh, cudaStream_t* mSplitKStreams, cudaEvent_t* mSplitKEvents, cublasHandle_t mCublas, cublasLtHandle_t mCublasLt, void **mWeights_d, float *mPostActivationScalesH_d, float *mPostActivationScalesY_d);

    IPluginV2IOExt* clone() const override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;

    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
    
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                 int nbOutputs, const DataType* inputTypes, const DataType* outputTypes,
                                 const bool* inputIsBroadcast, const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
    
    int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override;
    
    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;
    
    
    nvinfer1::DataType getOutputDataType (int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;
    
    bool isOutputBroadcastAcrossBatch (int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override;
    
    bool canBroadcastInputAcrossBatch (int inputIndex) const override;

    template <typename T>
    void write(char*& buffer, const T& val) const;

    template <typename T>
    void read(const char*& buffer, T& val) const;
    
private:
    int mAttentionSize;
    int mInputSize;
    
    int mHiddenSize;
    int mNumLayers;
    int mBeamSize;
    
    nvinfer1::DataType mDataType;
        
    cudaStream_t mStreami;
    cudaStream_t mStreamh;
    
    cudaStream_t* mSplitKStreams;
    cudaEvent_t* mSplitKEvents;
    
    cublasHandle_t mCublas;
    cublasLtHandle_t  mCublasLt;

    int mDevice;
    int mSMVersionMajor;
    int mSMVersionMinor;
    
    std::string mNamespace;
    
    void **mWeights_h;
    void **mWeights_d;
    
    float *mPostActivationScalesH_h;
    float *mPostActivationScalesH_d;
    
    float *mPostActivationScalesY_h;
    float *mPostActivationScalesY_d;
    
    float mLayerGemm0ScaleAttn;
    float mLayerGemm0ScaleInput;
};

class GNMTDecoderPluginCreator : public IPluginCreator {
public:
    GNMTDecoderPluginCreator() = default;

    ~GNMTDecoderPluginCreator() override = default;

    const char *getPluginName() const override;
    
    const char *getPluginVersion() const override;
    
    const PluginFieldCollection *getFieldNames() override;
    
    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;
    
    IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) override;
    
    IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // GNMT_DECODER_PLUGIN_H
