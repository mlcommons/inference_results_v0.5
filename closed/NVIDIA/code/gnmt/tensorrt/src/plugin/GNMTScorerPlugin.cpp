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

#include "GNMTScorerPlugin.h"
#include "GNMTScorerPluginKernel.h"
#include <numeric>
#include <functional>
#include <cassert>
#include <cstring>

// Clip plugin specific constants
namespace {
    static const char* GNMT_SCORER_PLUGIN_VERSION{"1.2"};
    static const char* GNMT_SCORER_PLUGIN_NAME{"GNMTScorerPlugin"};
}

// Static class fields initialization
PluginFieldCollection GNMTScorerPluginCreator::mFC{};
std::vector<PluginField> GNMTScorerPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GNMTScorerPluginCreator);

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

GNMTScorerPlugin::GNMTScorerPlugin(int topK, int eosIndex, int vocSize)
    : mTopK(topK)
    , mEosIndex(eosIndex)
    , mVocSize(vocSize)
{
}

GNMTScorerPlugin::GNMTScorerPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mTopK = readFromBuffer<int>(d);
    mEosIndex = readFromBuffer<int>(d);
    mBeamSize = readFromBuffer<int>(d);
    mVocSize = readFromBuffer<int>(d);
    mElemsPerRay = readFromBuffer<int>(d);
    mInputValType = static_cast<DataType>(readFromBuffer<int>(d));

    assert(d == (a + length));
}

const char* GNMTScorerPlugin::getPluginType() const
{
    return GNMT_SCORER_PLUGIN_NAME;
}

const char* GNMTScorerPlugin::getPluginVersion() const
{
    return GNMT_SCORER_PLUGIN_VERSION;
}

int GNMTScorerPlugin::getNbOutputs() const
{
    // 1) new combined logprobs and 2) selected tokens and indices inside the beam
    return 2;
}

Dims GNMTScorerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    auto lengthPenaltyDims = Dims{1, {2}};

    // Validate input arguments
    assert(nbInputDims == 4);
    assert(inputs[0].nbDims == 2);
    assert(std::accumulate(inputs[2].d, inputs[2].d + inputs[2].nbDims, 1, std::multiplies<int>()) == inputs[0].d[0]);
    assert(inputs[3] == lengthPenaltyDims);

    assert(index <= 1);

    // Both outputs has beamSize elements per sample
    return Dims{1, {mTopK}};
}

DataType GNMTScorerPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    assert(nbInputs == 4);

    if (index == 0)
        return DataType::kFLOAT;
    else if (index == 1)
        return DataType::kINT32;
    else
        assert(0);

    return DataType::kFLOAT;
}

bool GNMTScorerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool GNMTScorerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

int GNMTScorerPlugin::initialize()
{
    return 0;
}

void GNMTScorerPlugin::terminate()
{
}

size_t GNMTScorerPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int GNMTScorerPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    return runFinalReductionWithLengthPenalty(
        stream,
        batchSize,
        mTopK,
        mBeamSize,
        mElemsPerRay,
        mVocSize,
        mEosIndex,
        static_cast<const int *>(inputs[1]),
        inputs[0],
        mInputValType == DataType::kHALF,
        static_cast<const float *>(inputs[2]),
        static_cast<const float *>(inputs[3]),
        static_cast<float *>(outputs[0]),
        static_cast<int *>(outputs[1]));
}

size_t GNMTScorerPlugin::getSerializationSize() const
{
    return sizeof(int) * 6;
}

void GNMTScorerPlugin::serialize(void* buffer) const 
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mTopK);
    writeToBuffer(d, mEosIndex);
    writeToBuffer(d, mBeamSize);
    writeToBuffer(d, mVocSize);
    writeToBuffer(d, mElemsPerRay);
    writeToBuffer(d, static_cast<int>(mInputValType));

    assert(d == a + getSerializationSize());
}

bool GNMTScorerPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
{
    if (inOut[pos].format != TensorFormat::kNCHW)
        return false;

    switch(pos)
    {
        case 0:
            return (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
        case 1:
            return (inOut[pos].type == DataType::kINT32);
        case 2:
            return (inOut[pos].type == DataType::kFLOAT);
        case 3:
            return (inOut[pos].type == DataType::kFLOAT);
        case 4:
            return (inOut[pos].type == DataType::kFLOAT);
        case 5:
            return (inOut[pos].type == DataType::kINT32);
    }

    return false;
}

void GNMTScorerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
{
    auto targetOutputDims = Dims{1, {mTopK}};
    auto lengthPenaltyDims = Dims{1, {2}};

    assert(nbOutput == 2);
    assert(nbInput == 4);

    for(int i = 0; i < nbInput; ++i)
        assert(in[i].format == TensorFormat::kNCHW);
    for(int i = 0; i < nbOutput; ++i)
        assert(out[i].format == TensorFormat::kNCHW);

    assert(out[0].dims == targetOutputDims);
    assert(out[1].dims == targetOutputDims);
    assert(out[0].type == DataType::kFLOAT);
    assert(out[1].type == DataType::kINT32);

    assert(in[0].dims.nbDims == 2);
    assert(std::accumulate(in[2].dims.d, in[2].dims.d + in[2].dims.nbDims, 1, std::multiplies<int>()) == in[0].dims.d[0]);
    assert(in[3].dims == lengthPenaltyDims);
    assert((in[0].type == DataType::kFLOAT) || (in[0].type == DataType::kHALF));
    assert(in[1].type == DataType::kINT32);
    assert(in[2].type == DataType::kFLOAT);
    assert(in[3].type == DataType::kFLOAT);

    mBeamSize = in[0].dims.d[0]; 
    mElemsPerRay = in[0].dims.d[1]; 
    mInputValType = in[0].type;
}

void GNMTScorerPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    // delete this;
}

IPluginV2Ext* GNMTScorerPlugin::clone() const
{
    GNMTScorerPlugin* res = new GNMTScorerPlugin(mTopK, mEosIndex, mVocSize);
    res->mBeamSize = mBeamSize;
    res->mElemsPerRay = mElemsPerRay;
    res->mInputValType = mInputValType;

    return res;
}

void GNMTScorerPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GNMTScorerPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

GNMTScorerPluginCreator::GNMTScorerPluginCreator()
{
    // Describe ClipPlugin's required PluginField arguments
    mPluginAttributes.emplace_back(PluginField("topk", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eosindex", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("vocsize", nullptr, PluginFieldType::kINT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GNMTScorerPluginCreator::getPluginName() const
{
    return GNMT_SCORER_PLUGIN_NAME;
}

const char* GNMTScorerPluginCreator::getPluginVersion() const
{
    return GNMT_SCORER_PLUGIN_VERSION;
}

const PluginFieldCollection* GNMTScorerPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* GNMTScorerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int topK = 0;
    int eosIndex = 0;
    int vocSize = 0;
    const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 3);
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "topk") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            topK = *(static_cast<const int*>(fields[i].data));
        }
        else if (strcmp(fields[i].name, "eosindex") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            eosIndex = *(static_cast<const int*>(fields[i].data));
        }
        else if (strcmp(fields[i].name, "vocsize") == 0) {
            assert(fields[i].type == PluginFieldType::kINT32);
            vocSize = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new GNMTScorerPlugin(topK, eosIndex, vocSize);
}

IPluginV2* GNMTScorerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call GNMTScorerPlugin::destroy()
    return new GNMTScorerPlugin(serialData, serialLength);
}

void GNMTScorerPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* GNMTScorerPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
