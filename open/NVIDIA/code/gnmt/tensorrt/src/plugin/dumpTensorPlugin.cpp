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

#include "dumpTensorPlugin.h"

#include <cassert>
#include <cuda_runtime_api.h>
#include "common.h"
#include "utils.h"

namespace {
    static const char* GNMT_DUMP_TENSOR_PLUGIN_VERSION{"1.0"};
    static const char* GNMT_DUMP_TENSOR_PLUGIN_NAME{"GNMTDumpTensorPlugin"};
}

std::string DumpTensorPlugin::sDirName = "default_tensor_dump_dir";

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for serializing plugin
template <>
void writeToBuffer<string>(char*& buffer, const string& val){
    size_t length = val.length()+1;
    
    std::strcpy (buffer, val.c_str());
    const char * a = buffer;

    buffer += length * sizeof(char);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Helper function for deserializing plugin
std::string readStringFromBuffer(const char*& buffer, size_t length){
    char tmp[length];
    std::copy(buffer, buffer + length, tmp);
    const char * a = buffer;
    std::string val(tmp);
    buffer +=  length * sizeof(char); 

    return val;
}

// Static class fields initialization
PluginFieldCollection DumpTensorPluginCreator::mFC{};
std::vector<PluginField> DumpTensorPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DumpTensorPluginCreator);

DumpTensorPlugin::DumpTensorPlugin(const char * tensorName, bool isFp16)
    : mTensorName(tensorName)
    , mIsFp16(isFp16)
{
}

DumpTensorPlugin::DumpTensorPlugin(const void* data, size_t length)
{
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    size_t name_size = readFromBuffer<size_t>(d);
    mTensorName = readStringFromBuffer(d, name_size);
    mIsFp16 = readFromBuffer<bool>(d);
    mElementType = readFromBuffer<DataType>(d);
    mTensorVolume = readFromBuffer<int>(d);

    assert(d == a + getSerializationSize());
}

const char* DumpTensorPlugin::getPluginType() const
{
    return GNMT_DUMP_TENSOR_PLUGIN_NAME;
}

const char* DumpTensorPlugin::getPluginVersion() const
{
    return GNMT_DUMP_TENSOR_PLUGIN_VERSION;
}

int DumpTensorPlugin::getNbOutputs() const
{
    return 1;
}

Dims DumpTensorPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    return inputs[0];
}

DataType DumpTensorPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool DumpTensorPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool DumpTensorPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

int DumpTensorPlugin::initialize()
{
    return 0;
}

void DumpTensorPlugin::terminate()
{
}

size_t DumpTensorPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

bool DumpTensorPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (((mIsFp16 && (type == DataType::kHALF)) || ((!mIsFp16) && (type == DataType::kFLOAT))
           || (type == DataType::kINT32)) && (format == PluginFormat::kNCHW));
}

void DumpTensorPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    if (inputTypes[0] == DataType::kHALF)
        assert(mIsFp16);
    if (inputTypes[0] == DataType::kFLOAT)
        assert(!mIsFp16);
    mTensorVolume = samplesCommon::volume(inputDims[0]);
    mElementType = inputTypes[0];
}

int DumpTensorPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int totalElems = batchSize * mTensorVolume;
    cudaMemcpyAsync(outputs[0], inputs[0], totalElems * samplesCommon::getElementSize(mElementType), cudaMemcpyDeviceToDevice, stream);

    // Don't dump anything if inference is not started yet
    if (RuntimeInfo::currentBatch < 0)
        return 0;

    void * data;
    cudaHostAlloc(&data, mTensorVolume * batchSize * samplesCommon::getElementSize(mElementType), cudaHostAllocDefault);

    cudaMemcpyAsync(data, inputs[0], totalElems * samplesCommon::getElementSize(mElementType), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::string filename = mTensorName + "_b" + std::to_string(RuntimeInfo::currentBatch);

    switch (mElementType)
    {
        case DataType::kINT32:
            writeToUniqueFile(reinterpret_cast<const int32_t *>(data), totalElems, sDirName, filename);
            break;
        case DataType::kFLOAT:
            assert(!mIsFp16);
            writeToUniqueFile(reinterpret_cast<const float *>(data), totalElems, sDirName, filename);
            break;
        case DataType::kHALF:
            assert(mIsFp16);
            {
                half_float::half* hostPtr = reinterpret_cast<half_float::half*>(data);

                std::vector<float> hostFloatBuffer(totalElems);
                for (unsigned int i = 0; i < totalElems; i++)
                    hostFloatBuffer[i] = float(hostPtr[i]);
                writeToUniqueFile(&hostFloatBuffer[0], totalElems, sDirName, filename);
            }
            break;
        case DataType::kINT8:
            writeToUniqueFile(reinterpret_cast<const int8_t *>(data), totalElems, sDirName, filename);
            break;
        default:
            throw std::range_error("DataType is invalid.");
    }

    cudaFreeHost(data);

    return 0;
}

size_t DumpTensorPlugin::getSerializationSize() const
{
    return sizeof(size_t) + (mTensorName.length()+1) * sizeof(char) + sizeof(bool) + sizeof(DataType) + sizeof(int);
}

void DumpTensorPlugin::serialize(void* buffer) const
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    // Since mTensorName is a string of dynamic length, its size needs to be serialized as well
    writeToBuffer(d, mTensorName.length() + 1);
    writeToBuffer(d, mTensorName);
    writeToBuffer(d, mIsFp16);
    writeToBuffer(d, mElementType);
    writeToBuffer(d, mTensorVolume);

    assert(d == a + getSerializationSize());
}

void DumpTensorPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    // delete this;
}

IPluginV2Ext* DumpTensorPlugin::clone() const
{
    DumpTensorPlugin* res = new DumpTensorPlugin(mTensorName.c_str(), mIsFp16);
    res->mElementType = mElementType;
    res->mTensorVolume = mTensorVolume;

    return res;
}

void DumpTensorPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* DumpTensorPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

DumpTensorPluginCreator::DumpTensorPluginCreator()
{
}

const char* DumpTensorPluginCreator::getPluginName() const
{
    return GNMT_DUMP_TENSOR_PLUGIN_NAME;
}

const char* DumpTensorPluginCreator::getPluginVersion() const
{
    return GNMT_DUMP_TENSOR_PLUGIN_VERSION;
}

const PluginFieldCollection* DumpTensorPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* DumpTensorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    return new DumpTensorPlugin("", false);
}

IPluginV2* DumpTensorPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new DumpTensorPlugin(serialData, serialLength);
}

void DumpTensorPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* DumpTensorPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
