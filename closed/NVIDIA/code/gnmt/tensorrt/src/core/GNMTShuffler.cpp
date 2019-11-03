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

#include "GNMTShuffler.h"

#include "plugin/multiGatherPlugin.h"

Shuffler::Shuffler(std::shared_ptr<Config> config, bool profile):
    GNMTEngine("Shuffler", config, profile)
{
    // Initialize the buffer names.
    initBufferNames();
}

void Shuffler::initBufferNames()
{
    // Buffer name for parentBeamIndices.
    mParentBeamIndicesName = "Scorer_parent_beam_idx";
    // Buffer names for all the decoder states and the attention.
    for (int i = 0; i < mConfig->decoderLayerCount; i++)
    {
        mDecoderStateInputNames.emplace_back(std::string("cell_l" + std::to_string(i) + "_out"));
        mDecoderStateInputNames.emplace_back(std::string("hidd_l" + std::to_string(i) + "_out"));
        mDecoderStateOutputNames.emplace_back(std::string("cell_l" + std::to_string(i) + "_out_shuffled"));
        mDecoderStateOutputNames.emplace_back(std::string("hidd_l" + std::to_string(i) + "_out_shuffled"));
    }
    mDecoderStateInputNames.emplace_back("Attention_output");
    mDecoderStateOutputNames.emplace_back("Attention_output_shuffled");
}

void Shuffler::configureNetwork(nvinfer1::INetworkDefinition* network, int encoderMaxSeqLen)
{
    // Main input: Indices of previous beams for each new beam.
    ITensor* prevBeamIdxTensor = network->addInput(
        mParentBeamIndicesName.c_str(),
        DataType::kINT32,
        Dims{1, {mConfig->beamSize}, {DimensionType::kINDEX}});

    addDebugTensor(network, &prevBeamIdxTensor, mParentBeamIndicesName.c_str());

    // Add input layers for all the decoder states and the attention.
    vector<ITensor*> decoderStateTensors;
    DataType dataType = mConfig->prec;
    Dims2 dims(mConfig->beamSize, mConfig->hiddenSize);
    for (auto name : mDecoderStateInputNames)
    {
        auto inputTensor = addDecoderStateInput(network, name, dataType, dims);
        decoderStateTensors.push_back(inputTensor);
    }

    // Add gather layers.
    auto gatherLayerOutputs = addGatherLayers(network, prevBeamIdxTensor, decoderStateTensors);
    assert(gatherLayerOutputs.size() == mDecoderStateOutputNames.size());

    // Mark all the outputs.
    for (size_t i = 0; i < gatherLayerOutputs.size(); i++)
    {
        gatherLayerOutputs[i]->setName(mDecoderStateOutputNames[i].c_str());
        network->markOutput(*gatherLayerOutputs[i]);
    }
    setOutputDataType(network, mConfig->prec);
}

ITensor* Shuffler::addDecoderStateInput(nvinfer1::INetworkDefinition* network, const string& name, const DataType dataType, const Dims dims)
{
    auto inputTensor = network->addInput(name.c_str(), dataType, dims);
    assert(inputTensor);
    return inputTensor;
}

const vector<ITensor*> Shuffler::addGatherLayers(nvinfer1::INetworkDefinition* network, ITensor* parentBeamIndices, const vector<ITensor*>& decoderStateTensors)
{
    vector<ITensor*> gatherLayerOutputs;

    std::vector<ITensor *> pluginLayerInputs;
    for (auto decoderState : decoderStateTensors)
        pluginLayerInputs.push_back(decoderState);
    pluginLayerInputs.push_back(parentBeamIndices);

    MultiGatherPlugin plugin(decoderStateTensors.size(), mConfig->prec);
    auto pluginLayer = network->addPluginV2(&pluginLayerInputs.front(), pluginLayerInputs.size(), plugin);

    for(int i = 0; i < pluginLayer->getNbOutputs(); ++i)
        gatherLayerOutputs.push_back(pluginLayer->getOutput(i));

    return gatherLayerOutputs;
}

void Shuffler::setParentBeamIndicesInput(const void * data)
{
    setInputTensorBuffer("Scorer_parent_beam_idx", data);
}

void Shuffler::setAttentionInput(const void * data)
{
    setInputTensorBuffer("Attention_output", data);
}

void Shuffler::setHiddInput(int layerId, const void * data)
{
    setInputTensorBuffer((std::string("hidd_l") + std::to_string(layerId) + "_out").c_str(), data);
}

void Shuffler::setCellInput(int layerId, const void * data)
{
    setInputTensorBuffer((std::string("cell_l") + std::to_string(layerId) + "_out").c_str(), data);
}

std::shared_ptr<CudaBufferRaw> Shuffler::getAttentionShuffledOutput() const
{
    return getOutputTensorBuffer("Attention_output_shuffled");
}

std::shared_ptr<CudaBufferRaw> Shuffler::getHiddShuffledOutput(int layerId) const
{
    return getOutputTensorBuffer((std::string("hidd_l") + std::to_string(layerId) + "_out_shuffled").c_str());
}

std::shared_ptr<CudaBufferRaw> Shuffler::getCellShuffledOutput(int layerId) const
{
    return getOutputTensorBuffer((std::string("cell_l") + std::to_string(layerId) + "_out_shuffled").c_str());
}
