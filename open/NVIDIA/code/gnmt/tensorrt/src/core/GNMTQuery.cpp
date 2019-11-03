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

void Generator::importQueryWeights()
{
    // Load all weights from dump file
    std::vector<std::string> weightNames = {
        DEC_EMBED,
        DEC_LSTM_CELL0_BIAS,
        DEC_LSTM_CELL0_KERNEL,
    };
    mWeightsManager.importTFWeights(locateFile("gnmt.wts"), weightNames);
}

void Generator::addQuery(
    nvinfer1::INetworkDefinition* network,
    ITensor* indexTensor,
    ITensor* attnTensor,
    ITensor* hiddTensor,
    ITensor* cellTensor,
    ITensor** queryOutput,
    ITensor** hiddL0Out,
    ITensor** cellL0Out)
{
    // Add embedder and query
    auto decEmbedOutLayerOutput = addQueryEmbedLookup(network, indexTensor);
    auto queryOutLayer = addQuery(network, decEmbedOutLayerOutput, attnTensor, cellTensor, hiddTensor);

    // // Mark output tensor
    *queryOutput = queryOutLayer->getOutput(0);
    mConfig->addDebugTensor(network, queryOutput, "Query_output", "Query");
    (*queryOutput)->setName("Query_output");
    
    *hiddL0Out = queryOutLayer->getOutput(1);

    *cellL0Out = queryOutLayer->getOutput(2);    
}

ITensor* Generator::addQueryEmbedLookup(nvinfer1::INetworkDefinition* network, ITensor* input)
{
    auto weightsMap = mWeightsManager.getWeightsMap();

    // Add constant layer for embedding lookup table
    auto embeddingConstLayer = network->addConstant(
        Dims2{mConfig->vocabularySize, mConfig->hiddenSize},
        weightsMap[DEC_EMBED]);
    assert(embeddingConstLayer);

    embeddingConstLayer->setName("Query Embedding Layer");

    // Add gather layer
    auto gatherLayer = network->addGather(*embeddingConstLayer->getOutput(0), *input, 0);
    assert(gatherLayer);
    gatherLayer->setName("Gather Layer");

    auto gatherLayerOutput = gatherLayer->getOutput(0);
    mConfig->addDebugTensor(network, &gatherLayerOutput, "Query_embedder_output", "Query");

    return gatherLayerOutput;
}

ILayer* Generator::addQuery(nvinfer1::INetworkDefinition* network, ITensor* embeddOut, ITensor* oldAttnOut, ITensor* cellStates, ITensor* hiddenStates)
{
    //Add LSTM layer

    ITensor* inputTensors[] = {embeddOut, oldAttnOut};
    auto concatLstm = network->addConcatenation(inputTensors, 2);
    concatLstm->setAxis(2);
    concatLstm->setName("query_concatLstm");

    auto concatLstmOutput = concatLstm->getOutput(0);
    mConfig->addDebugTensor(network, &concatLstmOutput, "Query_lstm_input", "Query");

    auto lstm = network->addRNNv2(*concatLstmOutput, 1, mConfig->hiddenSize, 1, RNNOperation::kLSTM);
    assert(lstm);
    // lstm2->setInputMode(RNNInputMode::kLINEAR);
    // lstm2->setDirection(RNNDirection::kUNIDIRECTION);
    lstm->setCellState(*cellStates);
    lstm->setHiddenState(*hiddenStates);

    lstm->setName("query_LSTM");

    mWeightsManager.setTFRNNv2(lstm, vector<string>{DEC_LSTM_CELL0_KERNEL}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kWEIGHT);
    mWeightsManager.setTFRNNv2(lstm, vector<string>{DEC_LSTM_CELL0_BIAS}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kBIAS);

    return lstm;
}
