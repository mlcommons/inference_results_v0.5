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

#include "GNMTGenerator.h"

Generator::Generator(std::shared_ptr<Config> config, bool profile):
    GNMTEngine("Generator", config, profile)
{
}

void Generator::importWeights()
{
    importQueryWeights();
    importAttentionWeights();
    importDecoderWeights();
    importScorerWeights();
}

void Generator::configureNetwork(nvinfer1::INetworkDefinition* network, int encoderMaxSeqLen)
{
    auto indexTensor = addInput(network, "Query Embedding Indices", DataType::kINT32, Dims2{mConfig->beamSize, 1});
    assert(indexTensor);
    auto attnTensor = addInput(network, "Attention shuffled", mConfig->prec, Dims3{mConfig->beamSize, 1, mConfig->hiddenSize});
    assert(attnTensor);
    auto hiddTensor = addInput(network, "hidd_l0", mConfig->prec, Dims3{mConfig->beamSize, 1, mConfig->hiddenSize});
    assert(hiddTensor);
    auto cellTensor = addInput(network, "cell_l0", mConfig->prec, Dims3{mConfig->beamSize, 1, mConfig->hiddenSize});
    assert(cellTensor);

    ITensor * queryOutput;
    ITensor * hiddL0Out;
    ITensor * cellL0Out;
    addQuery(
        network,
        indexTensor,
        attnTensor,
        hiddTensor,
        cellTensor,
        &queryOutput,
        &hiddL0Out,
        &cellL0Out);
    assert(queryOutput);
    assert(hiddL0Out);
    assert(cellL0Out);

    hiddL0Out->setName("hidd_l0_out");
    network->markOutput(*hiddL0Out);
    hiddL0Out->setType(mConfig->prec);
    cellL0Out->setName("cell_l0_out");
    network->markOutput(*cellL0Out);
    cellL0Out->setType(mConfig->prec);
    


    auto encOut = addInput(network, "Encoder Output", mConfig->prec, Dims3{1, encoderMaxSeqLen, mConfig->hiddenSize});
    assert(encOut);
    auto encKeys = addInput(network, "Encoder Key Transform", mConfig->prec, Dims3{1, encoderMaxSeqLen, mConfig->hiddenSize});
    assert(encKeys);
    auto inputSeqLengths = addInput(network, "Input Sequence Lengths", DataType::kINT32, Dims2 {mConfig->beamSize, 1});
    assert(inputSeqLengths);

    ITensor * attentionOutput;
    addAttention(
        network,
        encOut,
        encKeys,
        queryOutput,
        inputSeqLengths,
        &attentionOutput,
        encoderMaxSeqLen);
    assert(attentionOutput);

    attentionOutput->setName("Attention_output");
    network->markOutput(*attentionOutput);
    attentionOutput->setType(mConfig->prec);

    std::vector<ITensor*> hiddStates;
    for (int i = 0; i < mConfig->decoderLayerCount - 1; ++i)
    {
        auto hiddTensor = addInput(network, std::string("hidd_l" + std::to_string(i + 1)).c_str(), mConfig->prec, Dims3{mConfig->beamSize, 1, mConfig->hiddenSize});
        assert(hiddTensor);
        hiddStates.push_back(hiddTensor);
    }
    std::vector<ITensor*> cellStates;
    for (int i = 0; i < mConfig->decoderLayerCount - 1; ++i)
    {
        auto cellTensor = addInput(network, std::string("cell_l" + std::to_string(i + 1)).c_str(), mConfig->prec, Dims3{mConfig->beamSize, 1, mConfig->hiddenSize});
        assert(cellTensor);
        cellStates.push_back(cellTensor);
    }

    std::vector<ITensor*> hiddStatesOut;
    std::vector<ITensor*> cellStatesOut;
    ITensor * decOut;
    if (mConfig->prec == DataType::kFLOAT || mConfig->isCalibrating())
    {
        addDecoder(
            network,
            queryOutput,
            attentionOutput,
            hiddStates,
            cellStates,
            hiddStatesOut,
            cellStatesOut,
            &decOut);
    }
    else
    {
        addDecoderPlugin(
            network,
            queryOutput,
            attentionOutput,
            hiddStates,
            cellStates,
            hiddStatesOut,
            cellStatesOut,
            &decOut,
            mConfig->prec);
    }

    for (int i = 0; i < mConfig->decoderLayerCount - 1; ++i)
        assert(hiddStatesOut[i]);
    for (int i = 0; i < mConfig->decoderLayerCount - 1; ++i)
        assert(cellStatesOut[i]);
    assert(decOut);

    for (int i = 0; i < mConfig->decoderLayerCount - 1; ++i)
    {
        hiddStatesOut[i]->setName(std::string("hidd_l" + std::to_string(i + 1) + "_out").c_str());
        network->markOutput(*hiddStatesOut[i]);
        hiddStatesOut[i]->setType(mConfig->prec);
    }
    for (int i = 0; i < mConfig->decoderLayerCount - 1; ++i)
    {
        cellStatesOut[i]->setName(std::string("cell_l" + std::to_string(i + 1) + "_out").c_str());
        network->markOutput(*cellStatesOut[i]);
        cellStatesOut[i]->setType(mConfig->prec);
    }

    auto parentLogProbs = addInput(network, "Scorer_parentLogProbs", nvinfer1::DataType::kFLOAT, Dims2{mConfig->beamSize, 1});
    assert(parentLogProbs);
    auto lengthPenaly = addInput(network, "Length_Penalty", DataType::kFLOAT, Dims{1, {2}});
    assert(lengthPenaly);

    ITensor* newCombinedLikelihoodsTensor;
    ITensor* newRayOptionIndicesTensor;
    addScorer(
        network,
        parentLogProbs,
        decOut,
        lengthPenaly,
        &newCombinedLikelihoodsTensor,
        &newRayOptionIndicesTensor);
    assert(newCombinedLikelihoodsTensor);
    assert(newRayOptionIndicesTensor);

    // Mark selected tokens and combined log probs as outputs, so that we can copy them to CPU.
    newCombinedLikelihoodsTensor->setName("logProbsCombined");
    network->markOutput(*newCombinedLikelihoodsTensor);
    newCombinedLikelihoodsTensor->setType(DataType::kFLOAT); // Set the output to fp32 regradless of math precision

    newRayOptionIndicesTensor->setName("Scorer_beamIndices");
    network->markOutput(*newRayOptionIndicesTensor);
    newRayOptionIndicesTensor->setType(DataType::kINT32);

}

void Generator::configureInt8Mode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network)
{

    // Create a calibrator (without BatchStreams)
    std::map <std::string, std::shared_ptr<BatchStream>> calibrationStreams;
    mCalibrator.reset(new Int8MinMaxCalibrator(calibrationStreams, 0, mName, mConfig->maxBatchSize, true, mConfig->calibrationCache));

    builder->setInt8Mode(true);

    builder->setInt8Calibrator(mCalibrator.get());

    /* 
    TensorScales tensorScales;
    mCalibrator->getScales(tensorScales);

    for(std::pair<std::string, float> scale : tensorScales){
        std::cout << scale.first << ": " << scale.second << std::endl;
    }
    
     */
}

void Generator::configureCalibrationCacheMode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network){
    std::map <std::string, std::shared_ptr<BatchStream>> calibrationStreams;

    for(int i = 0; i < network->getNbInputs(); i++){
        auto tensorName = (network->getInput(i))->getName();
        auto type = (network->getInput(i))->getType();
        calibrationStreams.insert(std::pair<std::string, std::shared_ptr<BatchStream>> (tensorName, std::shared_ptr<BatchStream> (new BatchStream(mConfig->maxBatchSize, mConfig->calibrationData, tensorName, type) ) ));
    }

    mCalibrator.reset(new Int8MinMaxCalibrator(calibrationStreams, 0, mName, mConfig->maxBatchSize, false, mConfig->calibrationCache));

    builder->setInt8Mode(true);

    builder->setInt8Calibrator(mCalibrator.get());
}

void Generator::setEmdeddingIndicesInput(const void * data)
{
    setInputTensorBuffer("Query Embedding Indices", data);
}

void Generator::setAttentionInput(const void * data)
{
    setInputTensorBuffer("Attention shuffled", data);
}

void Generator::setEncoderLSTMOutputInput(const void * data)
{
    setInputTensorBuffer("Encoder Output", data);
}

void Generator::setEncoderKeyTransformInput(const void * data)
{
    setInputTensorBuffer("Encoder Key Transform", data);
}

void Generator::setInputSequenceLengthsInput(const void * data)
{
    setInputTensorBuffer("Input Sequence Lengths", data);
}

void Generator::setParentLogProbsInput(const void * data)
{
    setInputTensorBuffer("Scorer_parentLogProbs", data);
}

void Generator::setLengthPenaltyInput(const void * data)
{
    setInputTensorBuffer("Length_Penalty", data);
}

void Generator::setHiddInput(int layerId, const void * data)
{
    setInputTensorBuffer((std::string("hidd_l") + std::to_string(layerId)).c_str(), data);
}

void Generator::setCellInput(int layerId, const void * data)
{
    setInputTensorBuffer((std::string("cell_l") + std::to_string(layerId)).c_str(), data);
}

std::shared_ptr<CudaBufferRaw> Generator::getAttentionOutput() const
{
    return getOutputTensorBuffer("Attention_output");
}

std::shared_ptr<CudaBufferRaw> Generator::getLogProbsCombinedOutput() const
{
    return getOutputTensorBuffer("logProbsCombined");
}

std::shared_ptr<CudaBufferRaw> Generator::getBeamIndicesOutput() const
{
    return getOutputTensorBuffer("Scorer_beamIndices");
}

std::shared_ptr<CudaBufferRaw> Generator::getHiddOutput(int layerId) const
{
    return getOutputTensorBuffer((std::string("hidd_l") + std::to_string(layerId) + "_out").c_str());
}

std::shared_ptr<CudaBufferRaw> Generator::getCellOutput(int layerId) const
{
    return getOutputTensorBuffer((std::string("cell_l") + std::to_string(layerId) + "_out").c_str());
}

