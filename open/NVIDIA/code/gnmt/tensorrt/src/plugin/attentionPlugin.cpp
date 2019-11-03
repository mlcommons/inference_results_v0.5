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

#include "attention.h"
#include "attentionPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

//#define DEBUG 0

using namespace nvinfer1;

namespace
{
const char* ATTENTION_PLUGIN_VERSION{"1"};
const char* ATTENTION_PLUGIN_NAME{"Attention_TRT"};
}

PluginFieldCollection AttentionPluginCreator::mFC{};
REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

//! Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

//! Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

AttentionPlugin::AttentionPlugin(std::string name, const void* data, size_t length)
    : mLayerName{name}
{
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    params.nbElements = readFromBuffer<int>(d);
    size_t nbDims = readFromBuffer<size_t>(d);
    for(int i = 0; i < 4; i++)
    {
        for(unsigned int j = 0; j < nbDims; j++)
        {
            params.inputStrides[i].push_back(readFromBuffer<int>(d));
        }
    }
    for(unsigned int j = 0; j < nbDims; j++)
    {
        params.resStrides.push_back(readFromBuffer<int>(d));
    }
    for(unsigned int j = 0; j < nbDims; j++)
    {
        params.len.push_back(readFromBuffer<int>(d));
    }
    params.type = (nvinfer1::DataType)(readFromBuffer<int>(d));
    assert(d == (a + length));
}

AttentionPlugin::AttentionPlugin(std::string name)
    : mLayerName{name}
{
}

//!
//! \brief This function computes the strides for a given tensor. If the length along a dimension is 1,
//!        the stride along that dimension is set to 0 to aid broadcast, else it is the product of the lower dimensions.
//!        Since dims.nbDims does not contain 
//!        the batch dimension, we add a placeholder for the batchStride at strides[0].
//!
std::vector<int> computeStrides(Dims dims, bool isBroadcastAccrossN)
{
    int stride = 1;
    std::vector<int> strides;
    strides.resize(dims.nbDims + 1);
    strides[dims.nbDims] = 1;
    if (dims.d[dims.nbDims - 1] == 1)
    {
        strides[dims.nbDims] = 0;
    }
    for (int i = dims.nbDims - 2; i >= 0; i--)
    {   
        stride *= dims.d[i + 1];        
        if (dims.d[i] == 1)
        {
            strides[i + 1] = 0;
        }
        else
        {
            strides[i + 1] = stride;
        }
    }
    strides[0] = isBroadcastAccrossN ? 0 : stride * dims.d[0];

#ifdef DEBUG
    for (int i : strides)
        std::cout << i << " ";
    std::cout << std::endl;
#endif
    return strides;
}

int AttentionPlugin::getNbOutputs() const
{
    return 1;
}

int AttentionPlugin::initialize()
{
    return 0;
}

void AttentionPlugin::terminate()
{
}

Dims AttentionPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index >= 0 && index < this->getNbOutputs());
    assert(inputs[0].nbDims == 3);
    assert(inputs[1].nbDims == 3);
    assert(inputs[2].nbDims == 3);
    assert(inputs[3].nbDims == 3);
    Dims outputDims;
    outputDims.nbDims = 3;
    
    outputDims.d[0] = inputs[1].d[0];
    outputDims.d[1] = inputs[0].d[1];
    outputDims.d[2] = 1;
    
    // printf("%d %d %d\n", outputDims.d[0], outputDims.d[1], inputs[0].d[2]);

    outputDims.type[0] = DimensionType::kCHANNEL;
    outputDims.type[1] = DimensionType::kSPATIAL;
    outputDims.type[2] = DimensionType::kSPATIAL;
    return outputDims;
}

size_t AttentionPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int AttentionPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    launchAttentionFusedKernel(params, batchSize, inputs, outputs[0], stream);
    return 0;
}

size_t AttentionPlugin::getSerializationSize() const
{
    size_t sz = 0;

    sz += sizeof(params.nbElements);
    sz += sizeof(params.inputStrides[0].size());
    
    //! Store the strides of the inputs
    for(int i = 0; i < 4; i++)
    {
        for(unsigned int j = 0; j < params.inputStrides[i].size(); j++)
        {
            sz += sizeof(params.inputStrides[i][j]);
        }
    }

    //! Store the strides of the output    
    for(unsigned int j = 0; j < params.resStrides.size(); j++)
    {
        sz += sizeof(params.resStrides[j]);
    }

    //! Store the lengths in each dimension        
    for(unsigned int j = 0; j < params.len.size(); j++)
    {
        sz += sizeof(params.len[j]);
    }
    
    sz += sizeof(int);
    
    return sz;
}

void AttentionPlugin::serialize(void* buffer) const
{
    //! Serialize each member of the AttentionParams struct
    char* d = static_cast<char*>(buffer);
    const char* a = d;
    writeToBuffer(d, params.nbElements);

    //! Store the number of dimensions to make reading/writing the strides easier
    writeToBuffer(d, params.inputStrides[0].size());   
    
    //! Store the strides of the inputs
    for(int i = 0; i < 4; i++)
    {
        for(unsigned int j = 0; j < params.inputStrides[i].size(); j++)
        {
            writeToBuffer(d, params.inputStrides[i][j]);
        }
    }

    //! Store the strides of the output    
    for(unsigned int j = 0; j < params.resStrides.size(); j++)
    {
        writeToBuffer(d, params.resStrides[j]);
    }

    //! Store the lengths in each dimension        
    for(unsigned int j = 0; j < params.len.size(); j++)
    {
        writeToBuffer(d, params.len[j]);
    }
    
    writeToBuffer(d, (int)params.type);
    
    assert(d == a + getSerializationSize());
}

void AttentionPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs == 4);
    assert(nbOutputs == 1);

    assert(format == PluginFormat::kNCHW);

    //! Compute Strides from inputDims and outputDims
    params.inputStrides[0] = computeStrides(inputDims[0], inputIsBroadcast[0]);
    params.inputStrides[1] = computeStrides(inputDims[1], inputIsBroadcast[1]);
    params.inputStrides[2] = computeStrides(inputDims[2], inputIsBroadcast[2]);
    params.inputStrides[3] = computeStrides(inputDims[3], inputIsBroadcast[3]);
    params.resStrides = computeStrides(outputDims[0], false);
    params.nbElements = 1;
    params.type = inputTypes[0];
    
    //! Add placeholder for batchSize dimension length
    params.len.push_back(1);


    //! Calculate number of elements in the output and the length of each dimension
    for (int i = 0; i < outputDims[0].nbDims; i++)
    {
        params.nbElements *= outputDims[0].d[i];
        params.len.push_back(outputDims[0].d[i]);
    }
#ifdef DEBUG
    std::cout << "Printing the lengths:\n";
    for (int i : params.len)
        std::cout << i << " ";
    std::cout << std::endl;
#endif
}

bool AttentionPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
}

const char* AttentionPlugin::getPluginType() const
{
    return ATTENTION_PLUGIN_NAME;
}

const char* AttentionPlugin::getPluginVersion() const
{
    return ATTENTION_PLUGIN_VERSION;
}

void AttentionPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* AttentionPlugin::clone() const
{
    auto* plugin = new AttentionPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->params = params;
    return plugin;
}

nvinfer1::DataType AttentionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool AttentionPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool AttentionPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    switch (inputIndex)
    {
    case 0: return true;
    case 1: return true;
    case 2: return true;
    case 3: return true;
    default: return false;
    }
    return false;
}

AttentionPluginCreator::AttentionPluginCreator()
{
}

const char* AttentionPluginCreator::getPluginName() const
{
    return ATTENTION_PLUGIN_NAME;
}

const char* AttentionPluginCreator::getPluginVersion() const
{
    return ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* AttentionPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* AttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    return new AttentionPlugin(name);
}

IPluginV2* AttentionPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new AttentionPlugin(name, serialData, serialLength);
}

void AttentionPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* AttentionPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
