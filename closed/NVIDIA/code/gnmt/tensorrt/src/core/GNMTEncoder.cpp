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

#include "GNMTEncoder.h"

Encoder::Encoder(std::shared_ptr<Config> config, bool profile):
    GNMTEngine("Encoder", config, profile)
{
}

void Encoder::importWeights()
{
    // Load all weights from dump file
    std::vector<std::string> weightNames = {
        ENC_EMBED,
        ENC_LSTM_BI_BW_BIAS, ENC_LSTM_BI_BW_KERNEL,
        ENC_LSTM_BI_FW_BIAS, ENC_LSTM_BI_FW_KERNEL,
        ENC_LSTM_CELL0_BIAS, ENC_LSTM_CELL0_KERNEL,
        ENC_LSTM_CELL1_BIAS, ENC_LSTM_CELL1_KERNEL,
        ENC_LSTM_CELL2_BIAS, ENC_LSTM_CELL2_KERNEL,
        ATTENTION_MEMORY_DENSE
        };
    mWeightsManager.importTFWeights(locateFile("gnmt.wts"), weightNames);
}

void Encoder::configureNetwork(nvinfer1::INetworkDefinition* network, int encoderMaxSeqLen)
{
    // Build model using the TensorRT API
    // Add inputs: Indices and Lengths
    auto indexTensor = network->addInput(
        "Encoder_embedding_indices",
        DataType::kINT32,
        Dims2{encoderMaxSeqLen, 1});
    assert(indexTensor);
    addDebugTensor(network, &indexTensor, "Encoder_embedding_indices");
    auto encoderSeqLenTensor = network->addInput(
        "Encoder Sequence Lengths",
        DataType::kINT32,
        Dims{1, {1}, {DimensionType::kCHANNEL}});
    assert(encoderSeqLenTensor);
    // Add embedder, encoder and key transform
    auto encEmbedOutLayerOutput = addEncEmbedLookup(network, indexTensor);

    auto encoderOutLayer = addEncoder(network, encEmbedOutLayerOutput, encoderSeqLenTensor, encoderMaxSeqLen);
    auto encoderOutLayerOutput = encoderOutLayer->getOutput(0);
    addDebugTensor(network, &encoderOutLayerOutput, "Encoder_lstm_output");

    auto keyTransformLayer = addKeyTransform(network, encoderOutLayerOutput);
    auto keyTransformLayerOutput = keyTransformLayer->getOutput(0);
    addDebugTensor(network, &keyTransformLayerOutput, "Encoder_key_transform");

    // Mark output tensor
    encoderOutLayerOutput->setName("Encoder_lstm_output");
    network->markOutput(*encoderOutLayerOutput);

    keyTransformLayerOutput->setName("Encoder_key_transform");
    network->markOutput(*keyTransformLayerOutput);


    setOutputDataType(network, mConfig->prec);
}

ILayer* Encoder::addKeyTransform(nvinfer1::INetworkDefinition* network, ITensor* input){
    auto weightsMap = mWeightsManager.getWeightsMap();
    auto encFC = network->addConstant(Dims3{1, mConfig->hiddenSize, mConfig->hiddenSize}, weightsMap[ATTENTION_MEMORY_DENSE]);
    encFC->setName("Key Transform Weights");

    auto encFCOut = network->addMatrixMultiply(*input, false, *encFC->getOutput(0), false);
    encFCOut->getOutput(0)->setName("Encoder Key Transform");

    return encFCOut;
}

ITensor* Encoder::addEncEmbedLookup(nvinfer1::INetworkDefinition* network, ITensor* input)
{
    auto weightsMap = mWeightsManager.getWeightsMap();

    // Add constant layer for embedding lookup table
    auto embeddingConstLayer = network->addConstant(
        Dims2{mConfig->vocabularySize, mConfig->hiddenSize},
        weightsMap[ENC_EMBED]);
    assert(embeddingConstLayer);

    // Add gather layer
    auto gatherLayer = network->addGather(*embeddingConstLayer->getOutput(0), *input, 0);
    assert(gatherLayer);
    gatherLayer->setName("Embed Gather Layer");
    auto gatherLayerOutput = gatherLayer->getOutput(0);

    addDebugTensor(network, &gatherLayerOutput, "Encoder_embedder_out");

    return gatherLayerOutput;
}

ILayer* Encoder::addEncoder(nvinfer1::INetworkDefinition* network, ITensor* input, ITensor* encoderSeqLenTensor, int encoderMaxSeqLen)
{

    Weights w;
    w.type = mConfig->prec;
    w.count = mConfig->hiddenSize * mConfig->beamSize;
    values = new float[mConfig->hiddenSize * mConfig->beamSize]();
    w.values = static_cast<void*> (values);
    auto zeroes = network->addConstant(Dims3{mConfig->beamSize, 1, mConfig->hiddenSize}, w);

    // Reshape the input dims to match the RNN dims
    auto reshapeLayer = network->addShuffle(*input);
    assert(reshapeLayer != nullptr);
    Dims inputDims = input->getDimensions();
    reshapeLayer->setReshapeDimensions(Dims3{inputDims.d[1], inputDims.d[0], inputDims.d[2]});
    reshapeLayer->getOutput(0)->setName("enc_embed_reshaped");

    std::vector<std::string> biLSTMWeights = {ENC_LSTM_BI_FW_KERNEL, ENC_LSTM_BI_BW_KERNEL};
    std::vector<std::string> biLSTMBiases = {ENC_LSTM_BI_FW_BIAS, ENC_LSTM_BI_BW_BIAS};
    auto biLstmLayer = addLSTM(network, reshapeLayer->getOutput(0), encoderSeqLenTensor, biLSTMWeights, biLSTMBiases, true, encoderMaxSeqLen);
    assert(biLstmLayer);
    auto biLstmLayerOutput0 = biLstmLayer->getOutput(0);

    addDebugTensor(network, &biLstmLayerOutput0, "biLstm_output");

    auto encL0HidBW = network->addSlice(*biLstmLayer->getOutput(1), Dims3{0, 1, 0}, Dims3{1, 1, 1024}, Dims3{1,1,1});

    // broadcast over mConfig->beamSize beams
    // Record hidden and cell state
    // TODO: broadcasting could be made more elegant by replacing the EW layer with another ISliceLayer:
    // using DimsCHW{0,1,1} for the stride.
    auto broadcastL0Hid = network->addElementWise(*encL0HidBW->getOutput(0), *zeroes->getOutput(0), ElementWiseOperation::kSUM);
    network->markOutput(*broadcastL0Hid->getOutput(0));
    broadcastL0Hid->getOutput(0)->setName("enc_l0_hid_BW");

    // TODO: similarly, the EW layer here can be replaced by another ISLice layer.
    auto encL0CellBW = network->addSlice(*biLstmLayer->getOutput(2), Dims3{0, 1, 0}, Dims3{1, 1, 1024}, Dims3{1,1,1});
    auto broadcastL0Cell = network->addElementWise(*encL0CellBW->getOutput(0), *zeroes->getOutput(0), ElementWiseOperation::kSUM);
    network->markOutput(*broadcastL0Cell->getOutput(0));
    broadcastL0Cell->getOutput(0)->setName("enc_l0_cell_BW");

    // Add all the unidirectional lstm layers
    std::vector<std::string> uniLSTMWeights = {
        ENC_LSTM_CELL0_KERNEL,
        ENC_LSTM_CELL1_KERNEL,
        ENC_LSTM_CELL2_KERNEL,
        ENC_LSTM_CELL3_KERNEL,
        ENC_LSTM_CELL4_KERNEL,
        ENC_LSTM_CELL5_KERNEL,
        ENC_LSTM_CELL6_KERNEL};
    std::vector<std::string> uniLSTMBiases = {
        ENC_LSTM_CELL0_BIAS,
        ENC_LSTM_CELL1_BIAS,
        ENC_LSTM_CELL2_BIAS,
        ENC_LSTM_CELL3_BIAS,
        ENC_LSTM_CELL4_BIAS,
        ENC_LSTM_CELL5_BIAS,
        ENC_LSTM_CELL6_BIAS};

    ILayer* uniLstmLayer;
    uniLstmLayer = addSingleLstmWithResidual(network, biLstmLayerOutput0, nullptr, encoderSeqLenTensor, zeroes->getOutput(0), {uniLSTMWeights[0], uniLSTMBiases[0]}, 1, encoderMaxSeqLen);
    for (int i = 1; i < mConfig->encoderLayerCount - 1; i++) {
        uniLstmLayer = addSingleLstmWithResidual(network, uniLstmLayer->getOutput(0), uniLstmLayer->getOutput(0), encoderSeqLenTensor, zeroes->getOutput(0), {uniLSTMWeights[i], uniLSTMBiases[i]}, i+1, encoderMaxSeqLen);
    }
    return uniLstmLayer;
}

ILayer* Encoder::addSingleLstmWithResidual(nvinfer1::INetworkDefinition* network, ITensor* input, ITensor* residualInput, ITensor* seqLenTensor, ITensor* zeroes, pair<string, string> WeightNames, int layerNum, int encoderMaxSeqLen)
{
    // Add the lstm layer.
    auto lstmLayer = addLSTM(network, input, seqLenTensor, vector<string>{WeightNames.first}, vector<string>{WeightNames.second}, false, encoderMaxSeqLen);
    assert(lstmLayer);

    // Make sure to record Hidden and cell state
    std::string name = std::string("enc_l") + std::to_string(layerNum) + std::string("_hid");
    auto broadcastHid = network->addElementWise(*lstmLayer->getOutput(1), *zeroes, ElementWiseOperation::kSUM);
    broadcastHid->getOutput(0)->setName(name.c_str());
    network->markOutput(*broadcastHid->getOutput(0));

    name = std::string("enc_l" + std::to_string(layerNum) + "_cell");
    auto broadcastCell = network->addElementWise(*lstmLayer->getOutput(2), *zeroes, ElementWiseOperation::kSUM);
    broadcastCell->getOutput(0)->setName(name.c_str());
    network->markOutput(*broadcastCell->getOutput(0));

    // Add residual connection if needed.
    if (residualInput) {
        auto* residualLayer = network->addElementWise(*lstmLayer->getOutput(0), *residualInput, ElementWiseOperation::kSUM);
        assert(residualLayer);
        return residualLayer;
    }

    return lstmLayer;
}

ILayer* Encoder::addLSTM(nvinfer1::INetworkDefinition* network, ITensor*  input, ITensor* seqLenTensor, std::vector<std::string> WeightNames, std::vector<std::string> BiasNames, bool isBi, int encoderMaxSeqLen)
{
    if (mConfig->isHalf() && (mConfig->maxBatchSize <= mConfig->persistentLSTMMaxBatchSize))
        return addPersistentLSTM(network, input, seqLenTensor, WeightNames, BiasNames, isBi);
    else
        return addTRTLSTM(network, input, seqLenTensor, WeightNames, BiasNames, isBi, encoderMaxSeqLen);
}

ILayer* Encoder::addTRTLSTM(nvinfer1::INetworkDefinition* network, ITensor*  input, ITensor* seqLenTensor, std::vector<std::string> WeightNames, std::vector<std::string> BiasNames, bool isBi, int encoderMaxSeqLen)
{
    // Add Bidirectional layer
    auto lstmLayer = network->addRNNv2(*input, 1, mConfig->hiddenSize, encoderMaxSeqLen, RNNOperation::kLSTM);
    assert(lstmLayer);
    if (isBi)
    {
        lstmLayer->setDirection(RNNDirection::kBIDIRECTION);
    }
    lstmLayer->setSequenceLengths(*seqLenTensor);

    // Set weights and bias for Bidirectional layer
    mWeightsManager.setTFRNNv2(lstmLayer, WeightNames, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kWEIGHT);
    mWeightsManager.setTFRNNv2(lstmLayer, BiasNames, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kBIAS);

    return lstmLayer;
}

ILayer* Encoder::addPersistentLSTM(nvinfer1::INetworkDefinition* network, ITensor*  input, ITensor* seqLenTensor, std::vector<std::string> WeightNames, std::vector<std::string> BiasNames, bool isBi)
{
    initLibNvInferPlugins(nullptr, "");
    auto creator = getPluginRegistry()->getPluginCreator("CgPersistentLSTMPlugin_TRT", "1");
    assert(creator);

    // set up basic info for the plugin lstm layer
    int inputSize = input->getDimensions().d[2];
    auto layerInfo = PersistentLSTMPluginInfo(inputSize, mConfig->hiddenSize, 1, isBi);
    int bidirectionFactor = layerInfo.isBi ? 2 : 1;

    // Setting up weights and bias
    Weights kernel;
    kernel.type = mConfig->prec;
    kernel.count = (inputSize + mConfig->hiddenSize) * 4 * mConfig->hiddenSize * bidirectionFactor;
    kernel.values = nullptr;

    mWeightsManager.setTFPersistentLSTM(WeightNames, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kWEIGHT, kernel, layerInfo);

    Weights bias;
    bias.type = mConfig->prec;
    bias.count = mConfig->hiddenSize * 4 * 2 * bidirectionFactor;
    bias.values = nullptr;
    mWeightsManager.setTFPersistentLSTM(BiasNames, TFCellType::kBASIC_LSTM_CELL, TFWeightsType::kBIAS, bias, layerInfo);

    Dims dimweights;
    dimweights.nbDims = 1;
    dimweights.d[0] = kernel.count;
    auto weightsConstLayer = network->addConstant(dimweights, kernel);

    Dims dimbias;
    dimbias.nbDims = 1;
    dimbias.d[0] = bias.count;
    auto biasConstLayer = network->addConstant(dimbias, bias);

    // set up inputs for the plugin layer
    std::vector<ITensor*> inputs;
    inputs.push_back(input);
    inputs.push_back(seqLenTensor);
    inputs.push_back(weightsConstLayer->getOutput(0));
    inputs.push_back(biasConstLayer->getOutput(0));
    inputs.push_back(nullptr);
    inputs.push_back(nullptr);

    int hiddenSize = mConfig->hiddenSize;
    int numLayers = 1;
    int setInitial = 0;
    PluginField fields[4] = {
        PluginField{"hiddenSize", &hiddenSize, PluginFieldType::kINT32, 1},
        PluginField{"numLayers", &numLayers, PluginFieldType::kINT32, 1},
        PluginField{"bidirectionFactor", &bidirectionFactor, PluginFieldType::kINT32, 1},
        PluginField{"setInitialStates", &setInitial, PluginFieldType::kINT32, 1},
    };

    PluginFieldCollection fc{4, fields};
    IPluginV2* cgPersistentLSTMPlugin = creator->createPlugin("CgPersistentLSTMPlugin_TRT", &fc);

    auto lstmLayer = network->addPluginV2(&inputs[0], 6, *cgPersistentLSTMPlugin);
    assert(lstmLayer);

    return lstmLayer;
}

void Encoder::setEmdeddingIndicesInput(const void * data)
{
    setInputTensorBuffer("Encoder_embedding_indices", data);
}

void Encoder::setInputSequenceLengthsInput(const void * data)
{
    setInputTensorBuffer("Encoder Sequence Lengths", data);
}

std::shared_ptr<CudaBufferRaw> Encoder::getLSTMOutput() const
{
    return getOutputTensorBuffer("Encoder_lstm_output");
}

std::shared_ptr<CudaBufferRaw> Encoder::getKeyTransformOutput() const
{
    return getOutputTensorBuffer("Encoder_key_transform");
}

std::shared_ptr<CudaBufferRaw> Encoder::getHiddOutput(int layerId) const
{
    if (layerId == 0)
        return getOutputTensorBuffer("enc_l0_hid_BW");
    else
        return getOutputTensorBuffer((std::string("enc_l") + std::to_string(layerId) + "_hid").c_str());
}

std::shared_ptr<CudaBufferRaw> Encoder::getCellOutput(int layerId) const
{
    if (layerId == 0)
        return getOutputTensorBuffer("enc_l0_cell_BW");
    else
        return getOutputTensorBuffer((std::string("enc_l") + std::to_string(layerId) + "_cell").c_str());
   
}
