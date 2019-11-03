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

#include "multiGatherPlugin.h"
#include "multiGatherPluginKernel.h"
#include "common.h"
#include <numeric>
#include <functional>
#include <cassert>
#include <cstring>

// Clip plugin specific constants
namespace {
    static const char* MULTI_GATHER_PLUGIN_VERSION{"1.0"};
    static const char* MULTI_GATHER_PLUGIN_NAME{"MultiGatherPlugin"};
}

// Static class fields initialization
PluginFieldCollection MultiGatherPluginCreator::mFC{};
std::vector<PluginField> MultiGatherPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MultiGatherPluginCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

inline bool operator ==(const Dims& x, const Dims& y)
{
    if (x.nbDims != y.nbDims)
        return false;
    for(int i = 0; i < x.nbDims; ++i)
        if (x.d[i] != y.d[i])
            return false;
    return true;
}

MultiGatherPlugin::MultiGatherPlugin(int tensorCount, DataType dataType)
    : mTensorCount(tensorCount)
    , mDataType(dataType)
{
    mVectorLengths.resize(mTensorCount);
    mSrcVectorCounts.resize(mTensorCount);
}

MultiGatherPlugin::MultiGatherPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mTensorCount = readFromBuffer<int>(d);
    mDataType = static_cast<DataType>(readFromBuffer<int>(d));
    mOutputVectorCount = readFromBuffer<int>(d);
    mVectorLengths.resize(mTensorCount);
    for(int i = 0; i < mTensorCount; ++i)
        mVectorLengths[i] = readFromBuffer<int>(d);
    mSrcVectorCounts.resize(mTensorCount);
    for(int i = 0; i < mTensorCount; ++i)
        mSrcVectorCounts[i] = readFromBuffer<int>(d);

    assert(d == (a + length));
}

const char* MultiGatherPlugin::getPluginType() const
{
    return MULTI_GATHER_PLUGIN_NAME;
}

const char* MultiGatherPlugin::getPluginVersion() const
{
    return MULTI_GATHER_PLUGIN_VERSION;
}

int MultiGatherPlugin::getNbOutputs() const
{
    return mVectorLengths.size();
}

Dims MultiGatherPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == (mTensorCount + 1));
    assert(inputs[mTensorCount].nbDims == 1);
    int outputVectorCount = inputs[mTensorCount].d[0];
    assert(index < mTensorCount);
    assert(inputs[index].nbDims == 2);
    int outputVectorLength = inputs[index].d[1];
    return Dims{2, {outputVectorCount, outputVectorLength}};
}

DataType MultiGatherPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    assert(nbInputs == (mTensorCount + 1));
    assert(index < mTensorCount);
    assert(inputTypes[index] == mDataType);
    return mDataType;
}

bool MultiGatherPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool MultiGatherPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

int MultiGatherPlugin::initialize()
{
    return 0;
}

void MultiGatherPlugin::terminate()
{
}

size_t MultiGatherPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int MultiGatherPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    return runMultiGather(
        stream,
        batchSize,
        mTensorCount,
        &mVectorLengths.front(),
        &mSrcVectorCounts.front(),
        mOutputVectorCount,
        samplesCommon::getElementSize(mDataType),
        inputs,
        outputs,
        static_cast<const int *>(inputs[mTensorCount]));
}

size_t MultiGatherPlugin::getSerializationSize() const
{
    return sizeof(int) * (mTensorCount * 2 + 3);
}

void MultiGatherPlugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mTensorCount);
    writeToBuffer(d, static_cast<int>(mDataType));
    writeToBuffer(d, mOutputVectorCount);
    for(auto elem: mVectorLengths)
        writeToBuffer(d, elem);
    for(auto elem: mSrcVectorCounts)
        writeToBuffer(d, elem);

    assert(d == a + getSerializationSize());
}

bool MultiGatherPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    if (inOut[pos].format != TensorFormat::kNCHW)
        return false;

    if (pos == mTensorCount)
        return (inOut[pos].type == DataType::kINT32);
    else
        return (inOut[pos].type == mDataType);

    return true;
}

void MultiGatherPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
{
    mOutputVectorCount = in[mTensorCount].dims.d[0];

    for(int i = 0; i < mTensorCount; ++i)
    {
        mSrcVectorCounts[i] = in[i].dims.d[0];
        mVectorLengths[i] = in[i].dims.d[1];
    }
}

void MultiGatherPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    // delete this;
}

IPluginV2Ext* MultiGatherPlugin::clone() const
{
    MultiGatherPlugin* res = new MultiGatherPlugin(mTensorCount, mDataType);
    res->mOutputVectorCount = mOutputVectorCount;
    res->mVectorLengths = mVectorLengths;
    res->mSrcVectorCounts = mSrcVectorCounts;

    return res;
}

void MultiGatherPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* MultiGatherPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

MultiGatherPluginCreator::MultiGatherPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("tensorcount", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("datatype", nullptr, PluginFieldType::kINT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MultiGatherPluginCreator::getPluginName() const
{
    return MULTI_GATHER_PLUGIN_NAME;
}

const char* MultiGatherPluginCreator::getPluginVersion() const
{
    return MULTI_GATHER_PLUGIN_VERSION;
}

const PluginFieldCollection* MultiGatherPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* MultiGatherPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int tensorCount = 0;
    int dataType = 0;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 1);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "tensorcount") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            tensorCount = *(static_cast<const int*>(fields[i].data));
        }
        else if (strcmp(fields[i].name, "datatype") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            dataType = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new MultiGatherPlugin(tensorCount, static_cast<DataType>(dataType));
}

IPluginV2* MultiGatherPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call GNMTScorerPlugin::destroy()
    return new MultiGatherPlugin(serialData, serialLength);
}

void MultiGatherPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* MultiGatherPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
