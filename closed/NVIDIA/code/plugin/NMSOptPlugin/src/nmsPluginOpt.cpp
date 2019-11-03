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

#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>

#include <cudnn.h>
#include "nmsPluginOpt.h"
#include "ssdOpt.h"
#include "ssdOptMacros.h"

using namespace nvinfer1;
using nvinfer1::plugin::NMSOptPluginCreator;
using nvinfer1::plugin::DetectionOutputOpt;
using nvinfer1::plugin::DetectionOutputParameters;

namespace
{
const char* NMS_OPT_PLUGIN_VERSION{"1"};
const char* NMS_OPT_PLUGIN_NAME{"NMS_OPT_TRT"};
}

PluginFieldCollection NMSOptPluginCreator::mFC{};
std::vector<PluginField> NMSOptPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(NMSOptPluginCreator);

DetectionOutputOpt::DetectionOutputOpt(DetectionOutputParameters params,
    bool confSoftmax, int numLayers)
    : param(params), mConfSoftmax(confSoftmax), mNumLayers(numLayers)
{
}

DetectionOutputOpt::DetectionOutputOpt(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<DetectionOutputParameters>(d);
    mConfSoftmax = read<bool>(d);
    mNumLayers = read<int>(d);
    C1 = read<int>(d);
    C2 = read<int>(d);
    numPriors = read<int>(d);
    mFeatureSize.resize(mNumLayers);
    mNumAnchors.resize(mNumLayers);
    for(int i = 0; i < mNumLayers; i++){
        mFeatureSize[i] = read<int>(d);
        mNumAnchors[i] = read<int>(d);
    }
    mPacked32NCHW = read<bool>(d);
    assert(d == a + length);
}

int DetectionOutputOpt::getNbOutputs() const
{
    return 1;
}

int DetectionOutputOpt::initialize()
{

    cudnnStatus_t status;
    status = cudnnCreate(&mCudnn);
    assert(status == CUDNN_STATUS_SUCCESS);
    status = cudnnCreateTensorDescriptor(&mInScoreTensorDesc);
    assert(status == CUDNN_STATUS_SUCCESS);
    status = cudnnCreateTensorDescriptor(&mOutScoreTensorDesc);
    assert(status == CUDNN_STATUS_SUCCESS);


    return 0;
}

void DetectionOutputOpt::terminate()
{
    cudnnStatus_t status;
    status = cudnnDestroyTensorDescriptor(mInScoreTensorDesc);
    assert(status == CUDNN_STATUS_SUCCESS);
    status = cudnnDestroyTensorDescriptor(mOutScoreTensorDesc);
    assert(status == CUDNN_STATUS_SUCCESS);
    status = cudnnDestroy(mCudnn);
    assert(status == CUDNN_STATUS_SUCCESS);
}

Dims DetectionOutputOpt::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert((nbInputDims-1)/2 == mNumLayers); //numInputs(box)+numInputs(conf)+prior
    assert(index == 0);
    mFeatureSize.resize(mNumLayers);
    mNumAnchors.resize(mNumLayers);

    C1 = 0;
    C2 = 0;

    const Dims* boxInputs = &inputs[param.inputOrder[0]];
    const Dims* confInputs = &inputs[param.inputOrder[1]];

    //both boxInputs and confInput are CHW
    for(int i = 0; i < mNumLayers; i++)
    {
        assert(boxInputs[i].nbDims == 3);
        assert(confInputs[i].nbDims == 3);
        int flattenBoxInput = boxInputs[i].d[0] * boxInputs[i].d[1] * boxInputs[i].d[2];
        int flattenConfInput = confInputs[i].d[0] * confInputs[i].d[1] * confInputs[i].d[2];
        C1 += flattenBoxInput, C2 += flattenConfInput;

        //same H and W for boxInputs and confInputs
        assert( boxInputs[i].d[1] == confInputs[i].d[1] && 
                boxInputs[i].d[2] == confInputs[i].d[2]);

        if(param.shareLocation) {
            assert( boxInputs[i].d[0] / 4 == confInputs[i].d[0] / param.numClasses );
        }
        else {
            assert( boxInputs[i].d[0] / 4 / param.numClasses == confInputs[i].d[0] / param.numClasses );
        }
        //Hack assert H=W
        assert(boxInputs[i].d[1] == boxInputs[i].d[2]);
        mFeatureSize[i] = boxInputs[i].d[1]; //assume H=W
        mNumAnchors[i] = boxInputs[i].d[0] / 4;
    }
    return DimsCHW(1, 1, param.keepTopK*7 + 1); //detections and keepCount
}

size_t DetectionOutputOpt::getWorkspaceSize(int maxBatchSize) const
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

int DetectionOutputOpt::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* const* locData = &inputs[param.inputOrder[0]];
    const void* const* confData = &inputs[param.inputOrder[1]];
    const void* priorData = inputs[param.inputOrder[2]];

    void* topDetections = outputs[0];

    ssdStatus_t status = detectionInferenceOpt(stream,
                                            batchSize,
                                            C1,
                                            C2,
                                            param.shareLocation,
                                            param.varianceEncodedInTarget,
                                            param.backgroundLabelId,
                                            numPriors,
                                            param.numClasses,
                                            param.topK,
                                            param.keepTopK,
                                            param.confidenceThreshold,
                                            param.nmsThreshold,
                                            param.codeType,
                                            DataType::kFLOAT,
                                            locData,
                                            priorData,
                                            DataType::kFLOAT,
                                            confData,
                                            topDetections,
                                            workspace,
                                            param.isNormalized,
                                            param.confSigmoid, 
                                            mConfSoftmax,
                                            mNumLayers,
                                            mFeatureSize.data(),
                                            mNumAnchors.data(),
                                            mPacked32NCHW,
                                            mCudnn,
                                            mInScoreTensorDesc,
                                            mOutScoreTensorDesc);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t DetectionOutputOpt::getSerializationSize() const
{
    // DetectionOutputParameters, mConfSoftmax, mNumLayers,C1,C2,numPriors
    return sizeof(DetectionOutputParameters) + sizeof(int) * 4 + sizeof(bool)
        //std::vector<int> mFeatureSize
        //std::vector<int> mNumAnchors
        + sizeof(int)*mNumLayers*2
        //mPacked32NCHW
        + sizeof(bool);
}

void DetectionOutputOpt::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, mConfSoftmax);
    write(d, mNumLayers);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    for(int i = 0; i < mNumLayers; i++) {
       write(d, mFeatureSize[i]);
       write(d, mNumAnchors[i]);
    }
    write(d, mPacked32NCHW);
    assert(d == a + getSerializationSize());
}

void DetectionOutputOpt::configurePlugin(const PluginTensorDesc* in, int nbInputs, const PluginTensorDesc* out, int nbOutputs)
{
    assert(out && nbOutputs == 1);
    assert(out[0].dims.nbDims == 3);

    assert(in && nbInputs == mNumLayers * 2 + 1); //mNumLayers each for conf/box + 1 for prior

    //prior
    assert(in[param.inputOrder[2]].dims.nbDims == 3);
    numPriors = in[param.inputOrder[2]].dims.d[1] / 4;

    const PluginTensorDesc* boxInputs = &in[param.inputOrder[0]];
    const PluginTensorDesc* confInputs = &in[param.inputOrder[1]];
    C1 = 0;
    C2 = 0;
    mFeatureSize.resize(mNumLayers);
    mNumAnchors.resize(mNumLayers);
    for(int i = 0; i < mNumLayers; i++) {
        const Dims &boxInputDims = boxInputs[i].dims;
        const Dims &confInputDims = confInputs[i].dims;
        assert(boxInputDims.nbDims == 3);
        assert(confInputDims.nbDims == 3);
        int flattenBoxInput = boxInputDims.d[0] * boxInputDims.d[1] * boxInputDims.d[2];
        int flattenConfInput = confInputDims.d[0] * confInputDims.d[1] * confInputDims.d[2];
        C1 += flattenBoxInput, C2 += flattenConfInput;

        //same H and W for boxInputDims and confInputDims
        assert( boxInputDims.d[1] == confInputDims.d[1] && 
                boxInputDims.d[2] == confInputDims.d[2]);

        if(param.shareLocation) {
            assert( boxInputDims.d[0] / 4 == confInputDims.d[0] / param.numClasses );
        }
        else {
            assert( boxInputDims.d[0] / 4 / param.numClasses == confInputDims.d[0] / param.numClasses );
        }

        //assert H=W
        assert(boxInputDims.d[1] == boxInputDims.d[2]);
        mFeatureSize[i] = boxInputDims.d[1]; //assert H=W
        mNumAnchors[i] = boxInputDims.d[0] / 4;

    }

    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
    assert(numPriors * numLocClasses * 4 == C1);
    assert(numPriors * param.numClasses == C2);

    //Check types and format
    for(int i = 0; i < mNumLayers; i++) {
        if(i == 0) {
            assert(boxInputs[i].type == DataType::kFLOAT && (boxInputs[i].format == TensorFormat::kCHW32 || boxInputs[i].format == TensorFormat::kLINEAR)); 
            assert(confInputs[i].type == DataType::kFLOAT && (confInputs[i].format == TensorFormat::kCHW32 || confInputs[i].format == TensorFormat::kLINEAR));
            assert(confInputs[i].format == boxInputs[i].format);

            if(boxInputs[i].format == TensorFormat::kCHW32)
                mPacked32NCHW = true;
            else
                mPacked32NCHW = false;
        }
        else {
            assert(boxInputs[i].type == DataType::kFLOAT && boxInputs[i].format == boxInputs[0].format); 
            assert(confInputs[i].type == DataType::kFLOAT && confInputs[i].format == confInputs[0].format);
        }

    }
    assert(in[param.inputOrder[2]].type == DataType::kFLOAT && in[param.inputOrder[2]].format == TensorFormat::kLINEAR);
    assert(out[0].type == DataType::kFLOAT && out[0].format == TensorFormat::kLINEAR);

}

bool DetectionOutputOpt::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const 
{
    assert(nbInputs == mNumLayers * 2 + 1); //mNumLayers each for conf/box + 1 for prior
    assert(nbOutputs == 1);

    bool rtn;
    if((pos >= param.inputOrder[0] && pos < param.inputOrder[0] + mNumLayers) //boxInput
        || (pos >= param.inputOrder[1] && pos < param.inputOrder[1] + mNumLayers)) //confInput
    {
        //use fp32 NC/32HW32
        rtn = inOut[pos].type == DataType::kFLOAT && (inOut[pos].format == TensorFormat::kCHW32 || inOut[pos].format == TensorFormat::kLINEAR);
        if(param.inputOrder[0] < param.inputOrder[1]) {
            rtn &= (inOut[pos].format == inOut[param.inputOrder[0]].format);
        }
        else {
            rtn &= (inOut[pos].format == inOut[param.inputOrder[1]].format);
        }
    }
    
    else if(pos == param.inputOrder[2]) //prior, just uses fp32 NCHW
        rtn = inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;

    else {
        assert(pos == nbInputs); // output
        rtn = inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    return rtn;
}
DataType DetectionOutputOpt::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const 
{
    assert(index == 0);
    assert(nbInputs == mNumLayers * 2 + 1);
    return DataType::kFLOAT;
}

const char* DetectionOutputOpt::getPluginType() const { return NMS_OPT_PLUGIN_NAME; }

const char* DetectionOutputOpt::getPluginVersion() const { return NMS_OPT_PLUGIN_VERSION; }

void DetectionOutputOpt::destroy() { delete this; }

IPluginV2IOExt* DetectionOutputOpt::clone() const
{
    IPluginV2IOExt* plugin = new DetectionOutputOpt(*this);
    return plugin;
}

bool DetectionOutputOpt::canBroadcastInputAcrossBatch(int inputIndex) const 
{
    if(inputIndex == param.inputOrder[2]) // prior
        return true;
    else
        return false;
}

NMSOptPluginCreator::NMSOptPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("varianceEncodedInTarget", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confSoftmax", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("codeType", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numLayers", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NMSOptPluginCreator::getPluginName() const
{
    return NMS_OPT_PLUGIN_NAME;
}

const char* NMSOptPluginCreator::getPluginVersion() const
{
    return NMS_OPT_PLUGIN_VERSION;
}

const PluginFieldCollection* NMSOptPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* NMSOptPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    //Default init values for TF SSD network
    params.codeType = CodeTypeSSD::TF_CENTER;
    params.inputOrder[0] = 0;
    params.inputOrder[1] = 7;
    params.inputOrder[2] = 6;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "varianceEncodedInTarget"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.varianceEncodedInTarget = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confidenceThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "nmsThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confSigmoid"))
        {
            params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confSoftmax"))
        {
            mConfSoftmax = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "inputOrder"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            const int* o = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.inputOrder[j] = *o;
                o++;
            }
        }
        else if (!strcmp(attrName, "codeType"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "numLayers"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mNumLayers = *(static_cast<const bool*>(fields[i].data));
        }

    }

    return new DetectionOutputOpt(params, mConfSoftmax, mNumLayers);
}

IPluginV2* NMSOptPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call NMS::destroy()
    return new DetectionOutputOpt(serialData, serialLength);
}
