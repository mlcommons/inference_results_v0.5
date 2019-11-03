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


void Generator::importDecoderWeights()
{
    std::vector<std::string> weightNames = {
        DEC_LSTM_CELL1_BIAS, DEC_LSTM_CELL1_KERNEL,
        DEC_LSTM_CELL2_BIAS, DEC_LSTM_CELL2_KERNEL,
        DEC_LSTM_CELL3_BIAS, DEC_LSTM_CELL3_KERNEL};

    mWeightsManager.importTFWeights(locateFile("gnmt.wts"), weightNames);
}

void Generator::addDecoder(
    INetworkDefinition* network,
    ITensor* decInput,
    ITensor* attnOut,
    const std::vector<ITensor*>& mHiddStates,
    const std::vector<ITensor*>& mCellStates,
    std::vector<ITensor*>& mHiddStatesOut,
    std::vector<ITensor*>& mCellStatesOut,
    ITensor** decOut)
{
    assert(static_cast<int>(mHiddStates.size()) == mConfig->decoderLayerCount - 1); // for (decoderLayerCount - 1) layers, from 1 to decoderLayerCount-1 inclusive
    assert(static_cast<int>(mCellStates.size()) == mConfig->decoderLayerCount - 1);

    mHiddStatesOut.resize(mConfig->decoderLayerCount - 1); // for (decoderLayerCount - 1) layers, from 1 to decoderLayerCount-1 inclusive
    mCellStatesOut.resize(mConfig->decoderLayerCount - 1);

    std::vector<IRNNv2Layer*> decLstm;
    std::vector<IElementWiseLayer*> decRes;

    // Add first layer
    ITensor* inputTensors[] = {decInput, attnOut};
    auto concatLstm1 = network->addConcatenation(inputTensors, 2);
    concatLstm1->setAxis(2);

    auto lstm1 = network->addRNNv2(*concatLstm1->getOutput(0), 1, mConfig->hiddenSize, 1, RNNOperation::kLSTM);
    assert(lstm1);
    lstm1->setInputMode(RNNInputMode::kLINEAR);
    lstm1->setDirection(RNNDirection::kUNIDIRECTION);
    lstm1->setCellState(*mCellStates[0]);
    lstm1->setHiddenState(*mHiddStates[0]);

    lstm1->setName("dec_LSTM_1");

    mHiddStatesOut[0] = lstm1->getOutput(1);
    mHiddStatesOut[0]->setName("hidd_l1_out");

    mCellStatesOut[0] = lstm1->getOutput(2);
    mCellStatesOut[0]->setName("cell_l1_out");

    mWeightsManager.setTFRNNv2(lstm1, vector<string>{DEC_LSTM_CELL1_KERNEL}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kWEIGHT);
    mWeightsManager.setTFRNNv2(lstm1, vector<string>{DEC_LSTM_CELL1_BIAS}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kBIAS);

    decLstm.push_back(lstm1);

    // Add second layer
    ITensor* inputTensors1[] = {decLstm[0]->getOutput(0), attnOut};
    auto concatLstm2 = network->addConcatenation(inputTensors1, 2);
    concatLstm2->setAxis(2);

    auto lstm2 = network->addRNNv2(*concatLstm2->getOutput(0), 1, mConfig->hiddenSize, 1, RNNOperation::kLSTM);
    assert(lstm2);
    lstm2->setInputMode(RNNInputMode::kLINEAR);
    lstm2->setDirection(RNNDirection::kUNIDIRECTION);
    lstm2->setCellState(*mCellStates[1]);
    lstm2->setHiddenState(*mHiddStates[1]);

    lstm2->setName("dec_LSTM_2");
    mHiddStatesOut[1] = lstm2->getOutput(1);
    mCellStatesOut[1] = lstm2->getOutput(2);

    mWeightsManager.setTFRNNv2(lstm2, vector<string>{DEC_LSTM_CELL2_KERNEL}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kWEIGHT);
    mWeightsManager.setTFRNNv2(lstm2, vector<string>{DEC_LSTM_CELL2_BIAS}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kBIAS);

    decLstm.push_back(lstm2);

    const std::vector<std::string> LSTMWeights = {DEC_LSTM_CELL3_KERNEL,
                                                  DEC_LSTM_CELL4_KERNEL,
                                                  DEC_LSTM_CELL5_KERNEL,
                                                  DEC_LSTM_CELL6_KERNEL,
                                                  DEC_LSTM_CELL7_KERNEL};
    const std::vector<std::string> LSTMBias = {DEC_LSTM_CELL3_BIAS,
                                               DEC_LSTM_CELL4_BIAS,
                                               DEC_LSTM_CELL5_BIAS,
                                               DEC_LSTM_CELL6_BIAS,
                                               DEC_LSTM_CELL7_BIAS};

    // Add layers > 2
    int i = 2;
    for (; i < mConfig->decoderLayerCount - 1; i++)
    {
        IElementWiseLayer* resCon;
        if (i == 2)
        {
            resCon = network->addElementWise(*decLstm[i - 2]->getOutput(0), *decLstm[i - 1]->getOutput(0), ElementWiseOperation::kSUM);
        }
        else
        {
            resCon = network->addElementWise(*decRes[i - 3]->getOutput(0), *decLstm[i - 1]->getOutput(0), ElementWiseOperation::kSUM);
        }
        assert(resCon);

        ITensor* tmpTensor = resCon->getOutput(0);
        ITensor* inputTensors2[] = {tmpTensor, attnOut};
        auto concatLstm1 = network->addConcatenation(inputTensors2, 2);
        concatLstm1->setAxis(2);
        auto tempLstm = network->addRNNv2(*concatLstm1->getOutput(0), 1, mConfig->hiddenSize, 1, RNNOperation::kLSTM);
        assert(tempLstm);

        tempLstm->setInputMode(RNNInputMode::kLINEAR);
        tempLstm->setDirection(RNNDirection::kUNIDIRECTION);
        tempLstm->setCellState(*mCellStates[i]);
        tempLstm->setHiddenState(*mHiddStates[i]);

        resCon->getOutput(0)->setName(std::string("Decoder_l" + std::to_string(i) + "_out").c_str());

        tempLstm->setName(std::string("LSTM_" + std::to_string(i)).c_str());
        resCon->setName(std::string("RES_" + std::to_string(i) + "_" + std::to_string(i + 1)).c_str());

        mHiddStatesOut[i] = tempLstm->getOutput(1);
        mCellStatesOut[i] = tempLstm->getOutput(2);

        mWeightsManager.setTFRNNv2(tempLstm, vector<string>{LSTMWeights[i - 2]}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kWEIGHT);
        mWeightsManager.setTFRNNv2(tempLstm, vector<string>{LSTMBias[i - 2]}, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kBIAS);

        decLstm.push_back(tempLstm);
        decRes.push_back(resCon);
    }

    IElementWiseLayer* recCon;
    recCon = network->addElementWise(*decRes[mConfig->decoderLayerCount - 4]->getOutput(0), *decLstm[mConfig->decoderLayerCount - 2]->getOutput(0), ElementWiseOperation::kSUM);
    assert(recCon);

    std::string sFinalOutput = std::string("Decoder_l" + std::to_string(i) + "_out");

    *decOut = recCon->getOutput(0);
    mConfig->addDebugTensor(network, decOut, sFinalOutput, "Decoder");

    (*decOut)->setName(sFinalOutput.c_str());

    decRes.push_back(recCon);
}

void Generator::addDecoderPlugin(
    nvinfer1::INetworkDefinition* network, 
    ITensor* decInput,
    ITensor* attnOut, 
    const std::vector<ITensor*> &mHiddStates, 
    const std::vector<ITensor*> &mCellStates,
    std::vector<ITensor*>& mHiddStatesOut,
    std::vector<ITensor*>& mCellStatesOut,
    ITensor** decOut,
    DataType prec)
{
    assert(static_cast<int>(mHiddStates.size()) == mConfig->decoderLayerCount - 1); // for (decoderLayerCount - 1) layers, from 1 to decoderLayerCount-1 inclusive
    assert(static_cast<int>(mCellStates.size()) == mConfig->decoderLayerCount - 1);

    mHiddStatesOut.resize(mConfig->decoderLayerCount - 1); // for (decoderLayerCount - 1) layers, from 1 to decoderLayerCount-1 inclusive
    mCellStatesOut.resize(mConfig->decoderLayerCount - 1);

    initLibNvInferPlugins(nullptr, "");
    auto creator = getPluginRegistry()->getPluginCreator(prec == DataType::kHALF ? "SingleStepLSTMPlugin" : "GNMTDecoderPlugin", "1");
    assert(creator);
    
    int numLayers = mConfig->decoderLayerCount - 1;

    const std::vector<std::string> LSTMWeights = {DEC_LSTM_CELL1_KERNEL,
                                                  DEC_LSTM_CELL2_KERNEL,
                                                  DEC_LSTM_CELL3_KERNEL,
                                                  DEC_LSTM_CELL4_KERNEL,
                                                  DEC_LSTM_CELL5_KERNEL,
                                                  DEC_LSTM_CELL6_KERNEL,
                                                  DEC_LSTM_CELL7_KERNEL};
    const std::vector<std::string> LSTMBias = {DEC_LSTM_CELL1_BIAS,
                                               DEC_LSTM_CELL2_BIAS,
                                               DEC_LSTM_CELL3_BIAS,
                                               DEC_LSTM_CELL4_BIAS,
                                               DEC_LSTM_CELL5_BIAS,
                                               DEC_LSTM_CELL6_BIAS,
                                               DEC_LSTM_CELL7_BIAS};
    
    std::vector<ITensor*> inputs;
    
    inputs.push_back(decInput);
    inputs.push_back(attnOut);
    
    for (int i = 0; i < numLayers; i++)
    {
        inputs.push_back(mHiddStates[i]);
    }
    for (int i = 0; i < numLayers; i++)
    {
        inputs.push_back(mCellStates[i]);
    }
    
    void **weightsData = (void**)malloc(numLayers * sizeof(void*));
    
    float **gemmOutputScaleValues = NULL;

    if (prec == DataType::kINT8) { 
        gemmOutputScaleValues = (float**)malloc(numLayers * sizeof(float*));
        
        for (int i = 0; i < numLayers; i++) {
            gemmOutputScaleValues[i] = (float*)malloc(12 * mConfig->hiddenSize * sizeof(float));
        }
    }
    
    auto weightsMap = mWeightsManager.getWeightsMap();
    
    for (int i = 0; i < numLayers; i++)
    {
        int inputSize;
        if (i == 0)
        {
            inputSize = decInput->getDimensions().d[2] + attnOut->getDimensions().d[2];
        }
        else
        {
            inputSize = mConfig->hiddenSize + attnOut->getDimensions().d[2];
        }
        
        // Ideally we wouldn't have this but there's a few places in the code where it's assumed to be true.
        assert(inputSize == 2 * mConfig->hiddenSize);
        
        nvinfer1::Weights weights = weightsMap[LSTMWeights[i]];
        
        // Get layer hyperparameters
        const int gateCount = 4;
        const int weightsCount = gateCount * mConfig->hiddenSize * (inputSize + mConfig->hiddenSize);

        if (weights.count != weightsCount)
        {
            string error = "Count of weights is not consistent with the expected count based on RNN hyperparameters.\n\n";
            error += "Expected Count: " + std::to_string(weightsCount) + "\n";
            error += "Observed Count: " + std::to_string(weights.count) + "\n";
            throw std::invalid_argument(error);
        }
        
        void *data = malloc(samplesCommon::getElementSize(weights.type) * weights.count);

        // Convert weights
        nvinfer1::Weights inputWeights{.type = weights.type, .values = weights.values, .count = gateCount*inputSize*mConfig->hiddenSize};
        int inWeightsDims[3]{inputSize, gateCount, mConfig->hiddenSize};
        int inWeightsOrder[3]{ 2, 0, 1};
        if (prec == DataType::kINT8) {
            memcpy(data, weights.values, gateCount*inputSize*mConfig->hiddenSize*samplesCommon::getElementSize(weights.type));
        }
        else 
        {
            nvinfer1::utils::reshapeWeights(inputWeights, inWeightsDims, inWeightsOrder, data, 3);
            nvinfer1::utils::transposeSubBuffers(data, weights.type, 1, inputSize * mConfig->hiddenSize, gateCount);
            
        }

        // Convert the recurrent weights
        void* recurValuesIn = (void *)((char *)weights.values + gateCount*inputSize*mConfig->hiddenSize*samplesCommon::getElementSize(weights.type));
        void* recurValuesOut = (void *)((char *)data + gateCount*inputSize*mConfig->hiddenSize*samplesCommon::getElementSize(weights.type));
        nvinfer1::Weights recurWeights{.type = weights.type, .values = recurValuesIn, .count = gateCount*mConfig->hiddenSize*mConfig->hiddenSize};
        int recurWeightsDims[3]{mConfig->hiddenSize, gateCount, mConfig->hiddenSize};
        int recurWeightsOrder[3]{ 2, 0, 1};
        if (prec == DataType::kINT8) {
            memcpy(recurValuesOut, recurValuesIn, gateCount*mConfig->hiddenSize*mConfig->hiddenSize*samplesCommon::getElementSize(weights.type));
        }
        else 
        {
            nvinfer1::utils::reshapeWeights(recurWeights, recurWeightsDims, recurWeightsOrder, recurValuesOut, 3);
            nvinfer1::utils::transposeSubBuffers(recurValuesOut, weights.type, 1, mConfig->hiddenSize * mConfig->hiddenSize, gateCount);
        }
        
        if (prec == DataType::kHALF) {
            half_float::half* dataHalf = (half_float::half*)malloc(weights.count * sizeof(half_float::half));
            
            for (int gate = 0; gate < gateCount; gate++) {
                // Swap from GNMT order to the order expected by the plugin
                int gateShuffle;
                if (gate == 1) {
                    gateShuffle = 2;
                }
                else if (gate == 2) {
                    gateShuffle = 1;
                }
                else {
                    gateShuffle = gate;
                }
                
                // Layer
                {            
                    int gateOffsetIn = gateShuffle * inputSize * mConfig->hiddenSize;
                    int gateOffsetOut = gate * inputSize * mConfig->hiddenSize;
                    
                    for (int j = 0; j < inputSize * mConfig->hiddenSize; j++) {
                        if (weights.type == DataType::kHALF) {
                            dataHalf[gateOffsetOut + j] = ((half_float::half*)data)[gateOffsetIn + j];
                        }
                        else {                        
                            dataHalf[gateOffsetOut + j] = half_float::half(((float*)(data))[gateOffsetIn + j]);
                        }
                    }
                }
                
                // Recurrent
                {            
                    int gateOffsetIn = gateCount*inputSize*mConfig->hiddenSize + gateShuffle * mConfig->hiddenSize * mConfig->hiddenSize;
                    int gateOffsetOut = gateCount*inputSize*mConfig->hiddenSize + gate * mConfig->hiddenSize * mConfig->hiddenSize;
                    
                    for (int j = 0; j < mConfig->hiddenSize * mConfig->hiddenSize; j++) {
                        if (weights.type == DataType::kHALF) {
                            dataHalf[gateOffsetOut + j] = ((half_float::half*)data)[gateOffsetIn + j];
                        }
                        else {                        
                            dataHalf[gateOffsetOut + j] = half_float::half(((float*)(data))[gateOffsetIn + j]);
                        }
                    }
                }
            }
            
            weightsData[i] = dataHalf;
            
            nvinfer1::Weights finalWeights{.type = DataType::kHALF, 
                                           .values = static_cast<const void *>(dataHalf), 
                                           .count = weights.count};
                                           
            auto weight = network->addConstant(Dims3{1, 1, 4 * mConfig->hiddenSize * (mConfig->hiddenSize + inputSize)}, finalWeights);
            inputs.push_back(weight->getOutput(0)); 

        }
        else if (prec == DataType::kINT8) {
            char* dataInt8 = (char*)malloc(weights.count * sizeof(char));
 
            // Scale and convert. Save scaling as that'll be needed to recover fp32 after the GEMM.
            for (int gate = 0; gate < gateCount; gate++) {
                // Swap from GNMT order to the order expected by the plugin
                int gateShuffle;
                if (gate == 1) {
                    gateShuffle = 2;
                }
                else if (gate == 2) {
                    gateShuffle = 1;
                }
                else {
                    gateShuffle = gate;
                }

                for (int row = 0; row < mConfig->hiddenSize; row++) {                
                    float maxAbsValL1 = 0;
                    float maxAbsValL2 = 0;
                    float maxAbsValR = 0;
                   
                    // Max Layer
                    {
                        int gateOffsetIn = gateShuffle * mConfig->hiddenSize;

                        for (int column = 0; column < mConfig->hiddenSize; column++) {
                            maxAbsValL1 = max((float)fabs(((float*)data)[gateOffsetIn + column * 4 * mConfig->hiddenSize + row]), maxAbsValL1);
                        }
                        
                        for (int column = mConfig->hiddenSize; column < 2 * mConfig->hiddenSize; column++) {
                            maxAbsValL2 = max((float)fabs(((float*)data)[gateOffsetIn + column * 4 * mConfig->hiddenSize + row]), maxAbsValL2);
                        }
                    }
                    
                    // Max Recurrent
                    {
                        int gateOffsetIn = gateCount*inputSize*mConfig->hiddenSize + gateShuffle * mConfig->hiddenSize;

                        for (int column = 0; column < mConfig->hiddenSize; column++) {
                            maxAbsValR = max((float)fabs(((float*)data)[gateOffsetIn + column * 4 * mConfig->hiddenSize + row]), maxAbsValR);
                        }
                    }
                    
                    // Save scaling factors. Multiply by this to go to floating point space from integer.
                    gemmOutputScaleValues[i][gate * mConfig->hiddenSize + row] = (maxAbsValL1 / 127.f);
                    gemmOutputScaleValues[i][4 * mConfig->hiddenSize + gate * mConfig->hiddenSize + row] = (maxAbsValL2 / 127.f);
                    gemmOutputScaleValues[i][8 * mConfig->hiddenSize + gate * mConfig->hiddenSize + row] = (maxAbsValR / 127.f);
                    
                    // Scale layer
                    {            
                        int gateOffsetIn = gateShuffle * mConfig->hiddenSize;
                        int gateOffsetOut = gate * mConfig->hiddenSize;
                        
                        for (int column = 0; column < mConfig->hiddenSize; column++) {
                            dataInt8[gateOffsetOut + column * 4 * mConfig->hiddenSize + row] = (int8_t)(127 * ((float*)data)[gateOffsetIn + column * 4 * mConfig->hiddenSize + row] / maxAbsValL1 + 0.5f);
                        }
                        
                        for (int column = mConfig->hiddenSize; column < 2 * mConfig->hiddenSize; column++) {
                            dataInt8[gateOffsetOut + column * 4 * mConfig->hiddenSize + row] = (int8_t)(127 * ((float*)data)[gateOffsetIn + column * 4 * mConfig->hiddenSize + row] / maxAbsValL2 + 0.5f);
                        }
                    }
                    
                    // Scale recurrent
                    {            
                        int gateOffsetIn = gateCount*inputSize*mConfig->hiddenSize + gateShuffle * mConfig->hiddenSize;
                        int gateOffsetOut = gateCount*inputSize*mConfig->hiddenSize + gate * mConfig->hiddenSize;
                        
                        for (int column = 0; column < mConfig->hiddenSize; column++) {
                            dataInt8[gateOffsetOut + column * 4 * mConfig->hiddenSize + row] = (int8_t)(127 * ((float*)data)[gateOffsetIn + column * 4 * mConfig->hiddenSize + row] / maxAbsValR + 0.5f);
                        }
                    }
                }
            }
            
            
            weightsData[i] = dataInt8;
        }
    }

    for (int i = 0; i < numLayers; i++)
    {
        nvinfer1::Weights biases = weightsMap[LSTMBias[i]];

        // Swap from GNMT order to the order expected by the plugin
        for (int j = 0; j < mConfig->hiddenSize; j++) {
            char* values = (char*)biases.values;
            
            char* tmp = (char*)malloc(samplesCommon::getElementSize(biases.type));
            
            memccpy(tmp, &values[(mConfig->hiddenSize + j) * samplesCommon::getElementSize(biases.type)], 1, samplesCommon::getElementSize(biases.type));            
            memccpy(&values[(mConfig->hiddenSize + j) * samplesCommon::getElementSize(biases.type)], &values[(2 * mConfig->hiddenSize + j) * samplesCommon::getElementSize(biases.type)], 1, samplesCommon::getElementSize(biases.type));
            memccpy(&values[(2 * mConfig->hiddenSize + j) * samplesCommon::getElementSize(biases.type)], tmp, 1, samplesCommon::getElementSize(biases.type));

            free(tmp);
        }            
        
        auto bias = network->addConstant(Dims3{1, 1, 4 * mConfig->hiddenSize}, biases);
        bias->setName(std::string("bias" + std::to_string(i)).c_str());
    
        inputs.push_back(bias->getOutput(0));
    }
    
    
    int numFields = 5 + numLayers + (prec == DataType::kINT8 ? 4 : 0);

    int dim = attnOut->getDimensions().d[2];
    PluginField *fields = (PluginField*)malloc(numFields * sizeof(PluginField));
    
    fields[0] = {"numLayers", &numLayers, PluginFieldType::kINT32, 1};
    fields[1] = {"hiddenSize", &mConfig->hiddenSize, PluginFieldType::kINT32, 1};
    fields[2] = {"attentionSize", &dim, PluginFieldType::kINT32, 1};
    fields[3] = {"beamSize", &mConfig->beamSize, PluginFieldType::kINT32, 1};
    fields[4] = {"dataType", &prec, PluginFieldType::kINT32, 1};
    
    // Weights
    for (int i = 0; i < numLayers; i++) {
        int inputSize;
        if (i == 0)
        {
            inputSize = decInput->getDimensions().d[2] + attnOut->getDimensions().d[2];
        }
        else
        {
            inputSize = mConfig->hiddenSize + attnOut->getDimensions().d[2];
        }
        
        PluginFieldType pluginFieldType;
        if (prec == DataType::kINT8) {
            pluginFieldType = PluginFieldType::kINT8;
        }
        else if (prec == DataType::kHALF) {
            pluginFieldType = PluginFieldType::kFLOAT16;
        }
        
        fields[5 + i] = {"weights", weightsData[i], pluginFieldType, 4 * mConfig->hiddenSize * (mConfig->hiddenSize + inputSize)};
    }

    
    if (prec == DataType::kINT8) {
        // Scales
        float *postActivationHValues = (float*)malloc(numLayers * sizeof(float));
        float *postActivationYValues = (float*)malloc(numLayers * sizeof(float));

        std::map <std::string, std::shared_ptr<BatchStream>> calibrationStreams;
        mCalibrator.reset(new Int8MinMaxCalibrator(calibrationStreams, 0, mName, mConfig->maxBatchSize, true, mConfig->calibrationCache));

        TensorScales tensorScales;
        mCalibrator->getScales(tensorScales);
        
        float layerGemm0ScaleAttn = 0;
        float layerGemm0ScaleInput = 0;
        
        for (int i = 0; i < numLayers; i++) {
            postActivationHValues[i] = -1;
            postActivationYValues[i] = -1;
        }

        for(std::pair<std::string, float> scale : tensorScales){
            if (!scale.first.compare("hidd_l1_out")) {
                postActivationHValues[0] = 1 / scale.second;
            }
            else if (!scale.first.compare("hidd_l2_out")) {
                postActivationHValues[1] = 1 / scale.second;
            }
            else if (!scale.first.compare("hidd_l3_out")) {
                postActivationHValues[2] = 1 / scale.second;
            }
            else if (!scale.first.compare("(Unnamed Layer* 15) [Concatenation]_output")) {
                postActivationYValues[0] = 1 / scale.second;
            }
            else if (!scale.first.compare("(Unnamed Layer* 18) [Concatenation]_output")) {
                postActivationYValues[1] = 1 / scale.second;
            }
            else if (!scale.first.compare("Decoder_l3_out")) {
                postActivationYValues[2] = 1 / scale.second;
            }
            else if (!scale.first.compare("Attention_output")) {
                layerGemm0ScaleAttn = 1 / scale.second;
            }
            else if (!scale.first.compare("Query_output")) {
                layerGemm0ScaleInput = 1 / scale.second;
            }
        }
                
        for (int i = 0; i < numLayers; i++) {
            assert(postActivationHValues[i] > 0);
            assert(postActivationYValues[i] > 0);
        }
        
        assert(layerGemm0ScaleAttn > 0);
        assert(layerGemm0ScaleInput > 0);

        
        fields[5 + numLayers] = {"postActivationHValues", postActivationHValues, PluginFieldType::kFLOAT32, numLayers};
        fields[5 + numLayers + 1] = {"postActivationYValues", postActivationYValues, PluginFieldType::kFLOAT32, numLayers};
        fields[5 + numLayers + 2] = {"layerGemm0ScaleInput", &layerGemm0ScaleInput, PluginFieldType::kFLOAT32, 1};
        fields[5 + numLayers + 3] = {"layerGemm0ScaleAttn", &layerGemm0ScaleAttn, PluginFieldType::kFLOAT32, 1};
    
    
        // Scales for the output of the GEMM from int32->fp32
        for (int i = 0; i < numLayers; i++)
        {
            
            for (int j = 0; j < 4 * mConfig->hiddenSize; j++) {
                if (i == 0) {
                    gemmOutputScaleValues[i][j] /= layerGemm0ScaleInput;
                }
                else {                    
                    gemmOutputScaleValues[i][j] /= postActivationYValues[i - 1];
                }
            }
            
            for (int j = 4 * mConfig->hiddenSize; j < 8 * mConfig->hiddenSize; j++) {
                if (i == 0) {
                    gemmOutputScaleValues[i][j] /= layerGemm0ScaleAttn;
                }
                else {                    
                    gemmOutputScaleValues[i][j] /= postActivationYValues[i - 1];
                }
            }
            
            for (int j = 8 * mConfig->hiddenSize; j < 12 * mConfig->hiddenSize; j++) {
                gemmOutputScaleValues[i][j] /= postActivationHValues[i];
            }
            
            nvinfer1::Weights gemmOutputScales{.type = DataType::kFLOAT, 
                                               .values = static_cast<const void *>(gemmOutputScaleValues[i]), 
                                               .count = 12 * mConfig->hiddenSize};

            auto scales = network->addConstant(Dims3{1, 1, 12 * mConfig->hiddenSize}, gemmOutputScales);
            
            scales->setName(std::string("gemmOutputScaleValues" + std::to_string(i)).c_str());
                        
            inputs.push_back(scales->getOutput(0));
        }
        
    }    
    free(weightsData);
    if (gemmOutputScaleValues) {
        free(gemmOutputScaleValues);
    }
    
   
    PluginFieldCollection fc{numFields, fields};

    IPluginV2* plugin = creator->createPlugin(prec == DataType::kHALF ? "SingleStepLSTMPlugin" : "GNMTDecoderPlugin", &fc);

    auto layer = network->addPluginV2(&inputs[0], int(inputs.size()), *plugin);
    layer->setName("Decoder pluginLayer");
    std::string sFinalOutput = std::string("Decoder_l" + std::to_string(numLayers) + "_out");
    layer->getOutput(0)->setName(sFinalOutput.c_str());
    assert(layer);
    
    
    *decOut = layer->getOutput(0);        

    for (int i = 0; i < numLayers; i++)
    {
        mHiddStatesOut[i] = layer->getOutput(i + 1);
        mCellStatesOut[i] = layer->getOutput(i + 1 + numLayers);
    }
}
