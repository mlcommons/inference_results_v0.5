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

#pragma once
#include "NvInferPlugin.h"
#include <cassert>
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class DetectionOutputOpt : public IPluginV2IOExt
{
public:
    DetectionOutputOpt(DetectionOutputParameters param, bool confSoftmax, int numLayers);

    DetectionOutputOpt(const void* data, size_t length);

    ~DetectionOutputOpt() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;
    
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2IOExt* clone() const override;

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override {return false;}

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

private:
    DetectionOutputParameters param;
    bool mConfSoftmax; 
    int mNumLayers;
    int C1, C2, numPriors;
    std::vector<int> mFeatureSize;
    std::vector<int> mNumAnchors;
    bool mPacked32NCHW;
    std::string mNamespace;
    cudnnHandle_t mCudnn;
    cudnnTensorDescriptor_t mInScoreTensorDesc;
    cudnnTensorDescriptor_t mOutScoreTensorDesc;
};

class NMSOptPluginCreator : public IPluginCreator
{
public:
    NMSOptPluginCreator();

    ~NMSOptPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    DetectionOutputParameters params;
    bool mConfSoftmax;
    int mNumLayers;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
