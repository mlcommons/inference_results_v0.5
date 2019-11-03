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

void Generator::importAttentionWeights()
{
    // Load all weights from dump file
    std::vector<std::string> weightNames = {ATTENTION_B, ATTENTION_G, ATTENTION_V, ATTENTION_MEMORY_DENSE, ATTENTION_QUERY_DENSE};
    mWeightsManager.importTFWeights(locateFile("gnmt.wts"), weightNames);

    auto weightsMap = mWeightsManager.getWeightsMap();
    // always creating float type of normScalVec. Also handling the case when ATTENTION_V is fp16.
    // TRT would internally convert weights to fp16 if needed (during build time, so no perf penalty)
    nvinfer1::Weights normScalVec{nvinfer1::DataType::kFLOAT, nullptr, 0};
    normScalVec.type = DataType::kFLOAT; 
    normScalVec.count = mConfig->hiddenSize;
    size_t numOfBytes = samplesCommon::getElementSize(normScalVec.type) * normScalVec.count;
    float* wtVals = static_cast<float*>(malloc(numOfBytes)); 
    float sq_sum = 0;
    float mult;
    if (weightsMap[ATTENTION_V].type == DataType::kFLOAT) {    
        auto attnV = static_cast<const float*>(weightsMap[ATTENTION_V].values);
        for (int i =0; i < normScalVec.count; i++) {
            sq_sum +=  attnV[i] * attnV[i];
        }
        sq_sum = sqrt(sq_sum);
        mult = static_cast<const float*>(weightsMap[ATTENTION_G].values)[0];
        for (int i = 0; i < normScalVec.count; i++) {
            wtVals[i] = attnV[i] / sq_sum;
            wtVals[i] = wtVals[i] * mult;
        }
    } else if (weightsMap[ATTENTION_V].type == DataType::kHALF) {
        cout<<"fp16 wts is experimental"<<endl;
        auto attnV = static_cast<const half_float::half*>(weightsMap[ATTENTION_V].values);
        for (int i =0; i < normScalVec.count; i++)
            sq_sum += float(attnV[i]) * float(attnV[i]);
        sq_sum = sqrt(sq_sum);
        mult = float( static_cast<const half_float::half*>(weightsMap[ATTENTION_G].values)[0] );
        for (int i = 0; i < normScalVec.count; i++) {
            wtVals[i] = float( attnV[i] ) / sq_sum;
            wtVals[i] = wtVals[i] * mult;
        }
    }
    else {
        cout<<"Data precision not supported for weights"<<endl;
    } 
    normScalVec.values = wtVals;
    mProcessedWeightsMap.insert(std::make_pair("attention_norm_scal_vec", normScalVec));
}

void Generator::addAttention(
    INetworkDefinition* network,
    ITensor* encOut,
    ITensor* encKeys,
    ITensor* decOut,
    ITensor* inputSeqLengths,
    ITensor** attentionOutput,
    int encoderMaxSeqLen)
{
    auto attentionOutLayer = addAttention(network, encOut, encKeys, decOut, inputSeqLengths, encoderMaxSeqLen);
    *attentionOutput = attentionOutLayer->getOutput(0);

    mConfig->addDebugTensor(network, attentionOutput, "Attention_output", "Attention");
}

ILayer* Generator::addAttention(
    nvinfer1::INetworkDefinition* network,
    ITensor* encOut,
    ITensor* encKeys,
    ITensor* decOut,
    ITensor* inputSeqLengths,
    int encoderMaxSeqLen)
{
    // Import Weights
    auto weightsMap = mWeightsManager.getWeightsMap();

    auto decFC = network->addConstant(Dims3{1, mConfig->hiddenSize, mConfig->hiddenSize}, weightsMap[ATTENTION_QUERY_DENSE]); 

    auto vecB = network->addConstant(Dims3{1, 1, mConfig->hiddenSize}, weightsMap[ATTENTION_B]);

    //std::cout<<"Using pre computed values for nsVecV"<<std::endl;
    auto nSVecV = network->addConstant( Dims3{1, mConfig->hiddenSize, 1}, mProcessedWeightsMap.find("attention_norm_scal_vec")->second );
    nSVecV->setName("Att: pre-computed NSVecV");


    // NETWORK

    auto decFCOut = network->addMatrixMultiply(*decOut, false, *decFC->getOutput(0), false);
    decFCOut->setName("Att: Decoder FC");

    auto decFCOutOutput = decFCOut->getOutput(0);
    mConfig->addDebugTensor(network, &decFCOutOutput, "Attention_processed_query", "Attention");


    auto creator = getPluginRegistry()->getPluginCreator("Attention_TRT", "1");
    auto *pluginObj = creator->createPlugin("fusedAttention", nullptr);

    
    ITensor* inputs[4] = {encKeys, decFCOutOutput, vecB->getOutput(0), nSVecV->getOutput(0)};
    auto layer = network->addPluginV2(&inputs[0], 4, *pluginObj);
    
    layer->setName("Attn: plugin");

    auto reshapeScores = network->addShuffle(*layer->getOutput(0));
    reshapeScores->setName("Att: reshapeScores");
    reshapeScores->setReshapeDimensions(Dims2{mConfig->beamSize, encoderMaxSeqLen});
    
    auto softmax = network->addRaggedSoftMax(*reshapeScores->getOutput(0), *inputSeqLengths);
    softmax->setName("Att: Alignment");
    // Outputting this layer with beam=5, seq=1, batch=1 is troublesome!

    auto reshapeSoftmax = network->addShuffle(*softmax->getOutput(0));
    reshapeSoftmax->setReshapeDimensions(Dims3{mConfig->beamSize, 1, encoderMaxSeqLen});
    reshapeSoftmax->setName("Att: Reshape");

    auto context = network->addMatrixMultiply(*reshapeSoftmax->getOutput(0), false, *encOut, false);
    context->setName("Att: context");

    return context;
}
