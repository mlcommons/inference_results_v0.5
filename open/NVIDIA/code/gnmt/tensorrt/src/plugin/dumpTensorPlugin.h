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

#ifndef GNMT_DUMP_TENSOR_PLUGIN_UTIL_
#define GNMT_DUMP_TENSOR_PLUGIN_UTIL_

#include "NvInferPlugin.h"

#include <list>
#include <memory>
#include <vector>

using namespace nvinfer1;

// To function correctly this plugin needs 2 things:
// 1) It should be essential part of the model, that is used eventually but at least one of the output.
//    Easy way to achieve it is to use its output whenever you would use original tensor (input), the plugin just copies input to output.
// 2) Runtime::current_batch >= 0, the plugin encodes batch # into filename.
class DumpTensorPlugin : public IPluginV2Ext
{
public:
    typedef std::shared_ptr<DumpTensorPlugin> ptr;

    DumpTensorPlugin(const char * tensorName, bool isFp16);

    DumpTensorPlugin(const void* data, size_t length);

    DumpTensorPlugin() = delete;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                         const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                         const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    static void setDirName(std::string dirName){sDirName = dirName; }

private:
    std::string mTensorName;
    bool mIsFp16;
    DataType mElementType;
    int mTensorVolume;
    std::string mNamespace;

    static std::string sDirName;
};

class DumpTensorPluginCreator : public IPluginCreator
{
public:
    DumpTensorPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    
    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // GNMT_DUMP_TENSOR_PLUGIN_UTIL_
