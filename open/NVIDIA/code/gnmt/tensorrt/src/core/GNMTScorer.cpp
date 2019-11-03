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

#include "plugin/GNMTScorerPlugin.h"
#include "common.h"
#include <algorithm>
#include <functional>



int32_t Generator::getPaddedVocSize(const Config& config)
{
    int32_t mScorerVocSize = config.vocabularySize;
    mScorerVocSize = samplesCommon::roundUp(mScorerVocSize, 32);
    return mScorerVocSize;
}

void Generator::importScorerWeights()
{
    // Load all weights from dump file
    std::vector<std::string> weightNames = {DECODE_DENSE_KERNEL};
    mWeightsManager.importTFWeights(locateFile("gnmt.wts"), weightNames);

    // Pad the vocabulary so we can call HMMA/IMMA kernels on the projection layer.
    resizeScorerWeights(
        DECODE_DENSE_KERNEL,
        mConfig->hiddenSize,
        mConfig->vocabularySize,
        mConfig->hiddenSize,
        getPaddedVocSize(*mConfig));
}

std::pair<Weights, Weights> Generator::getConvolutionWeightsAndBias()
{
    auto convWeightsOrg = mProcessedWeightsMap.find("scorer_padded_weights")->second;
    auto convBias = mProcessedWeightsMap.find("scorer_padded_bias")->second;

    // Transpose weights
    void* values = const_cast<void*>(convWeightsOrg.values);
    nvinfer1::utils::transposeSubBuffers(values, DataType::kFLOAT, 1, mConfig->hiddenSize, getPaddedVocSize(*mConfig));

    Weights convWeights{.type = DataType::kFLOAT, .values = values, .count = convWeightsOrg.count};

    return make_pair(convWeights, convBias);
}

void Generator::resizeScorerWeights(
    const std::string name,
    const unsigned int inputRows,
    const unsigned int inputCols,
    const unsigned int outputRows,
    const unsigned int outputCols)
{
    auto weightsMap = mWeightsManager.getWeightsMap();
    assert(weightsMap[name].count == inputRows * inputCols);
    assert(inputRows <= outputRows && inputCols <= outputCols);
    auto input = static_cast<const float*>((weightsMap[name]).values);
    float* resizedWeights = static_cast<float*>(malloc(outputRows * outputCols * sizeof(float)));
    std::fill_n(resizedWeights, outputRows * outputCols, 0.0F);

    for (unsigned int r = 0; r < inputRows; ++r)
    {
        for (unsigned int c = 0; c < inputCols; c++)
        {
            resizedWeights[r * outputCols + c] = input[r * inputCols + c];
        }
    }
    nvinfer1::Weights mWeights;
    mWeights.type = nvinfer1::DataType::kFLOAT;
    mWeights.values = resizedWeights;
    mWeights.count = outputRows * outputCols;
    mProcessedWeightsMap.insert(std::make_pair("scorer_padded_weights", mWeights));

    // Assign lowest value to the padded part to mask out those vocabulary.
    nvinfer1::Weights mBias;
    float* biasBuf = static_cast<float*>(malloc(outputCols * sizeof(float)));
    std::fill_n(biasBuf, inputCols, 0.0F);
    std::fill_n(biasBuf + inputCols, outputCols - inputCols, -std::numeric_limits<float>::infinity());
    mBias.type = nvinfer1::DataType::kFLOAT;
    mBias.values = biasBuf;
    mBias.count = outputCols;
    mProcessedWeightsMap.insert(std::make_pair("scorer_padded_bias", mBias));
}

void Generator::addScorer(
    INetworkDefinition* network,
    ITensor* parentLogProbs,
    ITensor* decoderOutput,
    ITensor* lengthPenaly,
    ITensor** newCombinedLikelihoodsTensor,
    ITensor** newRayOptionIndicesTensor)
{
    ITensor* softmaxInput{nullptr};

    if (mConfig->useInt8ProjectionGraph())
    {
        auto reshapeDecoderOutputLayer = network->addShuffle(*decoderOutput);
        reshapeDecoderOutputLayer->setName("Scorer reshapeDecoderOutputLayer");
        reshapeDecoderOutputLayer->setReshapeDimensions(DimsNCHW{mConfig->beamSize, mConfig->hiddenSize, 1, 1});
        auto decoderOutputReshaped = reshapeDecoderOutputLayer->getOutput(0);

        // Prepare Conv weights and bias
        auto WeightsAndBias = getConvolutionWeightsAndBias();

        // Projection
        // Matrix Multiply is implemented by 1x1 convolution layer. Current implementation of MatrixMultiply Layer does not have
        // Int8 inference support, so we rely on convolution to invoke gemm operation required for projection.
        auto projectionLayer = network->addConvolution(*decoderOutputReshaped, getPaddedVocSize(*mConfig), DimsHW(1, 1), WeightsAndBias.first, WeightsAndBias.second);
        projectionLayer->setName("Int8ProjectionLayer");
        assert(projectionLayer);
        projectionLayer->setOutputType(0, nvinfer1::DataType::kFLOAT);

        // Force the Shuffle layer to be FP32 so that we can take advantage of in-place reformat and in-place shuffle.
        auto dimEqualizeLayer = network->addShuffle(*projectionLayer->getOutput(0));
        assert(dimEqualizeLayer);
        dimEqualizeLayer->setReshapeDimensions(Dims2{mConfig->beamSize, getPaddedVocSize(*mConfig)});
        dimEqualizeLayer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        dimEqualizeLayer->setName("Scorer dimEqualizeLayer");

        softmaxInput = dimEqualizeLayer->getOutput(0);
        mConfig->addDebugTensor(network, &softmaxInput, "Scorer_logits", "Scorer");
    }
    else
    {
        auto reshapeDecoderOutputLayer = network->addShuffle(*decoderOutput);
        reshapeDecoderOutputLayer->setReshapeDimensions(Dims2{mConfig->beamSize, mConfig->hiddenSize});
        auto decoderOutputReshaped = reshapeDecoderOutputLayer->getOutput(0);

        auto constantLayer = network->addConstant(Dims2{mConfig->hiddenSize, getPaddedVocSize(*mConfig)}, mProcessedWeightsMap.find("scorer_padded_weights")->second);
        assert(constantLayer);
        auto projectionLayer = network->addMatrixMultiply(*decoderOutputReshaped, false, *constantLayer->getOutput(0), false);
        assert(projectionLayer);
        projectionLayer->setName("Projection Matrix Multiply");

        auto biasLayer = network->addConstant(Dims2{1, getPaddedVocSize(*mConfig)}, mProcessedWeightsMap.find("scorer_padded_bias")->second);
        assert(biasLayer);
        auto maskLayer = network->addElementWise(*projectionLayer->getOutput(0), *biasLayer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        assert(maskLayer);

        softmaxInput = maskLayer->getOutput(0);
        mConfig->addDebugTensor(network, &softmaxInput, "Scorer_logits", "Scorer");
    }

    ITensor* intermediateLogprobsTensor{nullptr};
    ITensor* intermediateIndicesTensor{nullptr};
    
    ITensor* softmaxOutput{nullptr};
    auto softmaxLayer = network->addSoftMax(*softmaxInput);
    softmaxLayer->setName("Scorer softmaxLayer");
    assert(softmaxLayer);
    softmaxLayer->setAxes(2);
    softmaxOutput = softmaxLayer->getOutput(0);
    auto logLayer = network->addUnary(*softmaxOutput, UnaryOperation::kLOG);
    logLayer->setName("Scorer logLayer");
    assert(logLayer);
    auto topKLayer = network->addTopK(*logLayer->getOutput(0), TopKOperation::kMAX, mConfig->beamSize + 1, 2);
    topKLayer->setName("Scorer topKLayer");
    assert(topKLayer);
    intermediateLogprobsTensor = topKLayer->getOutput(0);
    intermediateIndicesTensor = topKLayer->getOutput(1);

    assert(intermediateLogprobsTensor);
    assert(intermediateIndicesTensor);

    ITensor * pluginLayerInputs[] = {intermediateLogprobsTensor, intermediateIndicesTensor, parentLogProbs, lengthPenaly};
    GNMTScorerPlugin plugin(mConfig->beamSize, mConfig->STOP_TOKEN, mConfig->vocabularySize);
    auto pluginLayer = network->addPluginV2(pluginLayerInputs, 4, plugin);
    pluginLayer->setName("Scorer pluginLayer");
    assert(pluginLayer);
    *newCombinedLikelihoodsTensor = pluginLayer->getOutput(0);
    assert(*newCombinedLikelihoodsTensor);
    *newRayOptionIndicesTensor = pluginLayer->getOutput(1);
    assert(*newRayOptionIndicesTensor);
}
