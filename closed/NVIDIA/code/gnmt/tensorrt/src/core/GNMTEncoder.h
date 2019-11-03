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

#ifndef GNMT_ENCODER_H
#define GNMT_ENCODER_H

#include "GNMTEngine.h"
#include "params.h"

using namespace nvinfer1;

class Encoder : public GNMTEngine
{
public:
    Encoder();

    Encoder(std::shared_ptr<Config> config, bool profile);

    ~Encoder() { delete values; }

    void setEmdeddingIndicesInput(const void * data);

    void setInputSequenceLengthsInput(const void * data);

    std::shared_ptr<CudaBufferRaw> getLSTMOutput() const;

    std::shared_ptr<CudaBufferRaw> getKeyTransformOutput() const;

    std::shared_ptr<CudaBufferRaw> getHiddOutput(int layerId) const;

    std::shared_ptr<CudaBufferRaw> getCellOutput(int layerId) const;


private:
    //!
    //! \brief Imports weights specific fo the embedder and encoder.
    //!
    void importWeights();

    //!
    //! \brief Configures the network to add the encoder embedded lookup and the encoder LSTM layers.
    //!
    void configureNetwork(nvinfer1::INetworkDefinition* network, int encoderMaxSeqLen);

    //!
    //! \brief Adds the encoder embedded lookup portion of GNMT to the network.
    //!
    //! \note This is being implemented using a const layer to contain the lookup table
    //!       and a gather layer to perform the lookup in parallel for multiple token indicies.
    //!
    ITensor* addEncEmbedLookup(nvinfer1::INetworkDefinition* network, ITensor* input);

    //!
    //! \brief Adds the encoder LSTM portion of GNMT to the network.
    //!
    //! \note This is being implemented using using the RNNv2 and Elementwise layers.
    //!           1) The RNNv2 layers is used to represent both the bidirectional and unidirectional RNN layers.
    //!           2) The Elementwise layer is used to add previous RNN outputs to produce the residual input required
    //!              by the GNMT network.
    //!
    ILayer* addEncoder(nvinfer1::INetworkDefinition* network, ITensor* input, ITensor* encoderSeqLenTensor, int encoderMaxSeqLen);

    ILayer* addKeyTransform(nvinfer1::INetworkDefinition*, nvinfer1::ITensor*);
    //!
    //! \brief Adds a single LSTM layer with optional residual connection to the encoder network.
    //!
    //! \param network The encoder network.
    //! \param input The input tensor to the LSTM layer.
    //! \param residualInput The tensor that will be added to the output from the LSTM layer. Can be the same as input.
    //!        Pass nullptr if residual is not needed.
    //! \param seqLenTensor The tensor containing the sequence length.
    //! \param WeightNames A tuple containing the names of the weights and the bias.
    //!
    ILayer* addSingleLstmWithResidual(nvinfer1::INetworkDefinition* network, ITensor* input, ITensor* residualInput, ITensor* seqLenTensor, ITensor* zeroes, pair<string, string> WeightNames, int layerNum, int encoderMaxSeqLen);

    ILayer* addLSTM(nvinfer1::INetworkDefinition* network, ITensor*  input, ITensor* seqLenTensor, std::vector<std::string> WeightNames, std::vector<std::string> BiasNames, bool isBi, int encoderMaxSeqLen);
    ILayer* addTRTLSTM(nvinfer1::INetworkDefinition* network, ITensor*  input, ITensor* seqLenTensor, std::vector<std::string> WeightNames, std::vector<std::string> BiasNames, bool isBi, int encoderMaxSeqLen);
    ILayer* addPersistentLSTM(nvinfer1::INetworkDefinition* network, ITensor*  input, ITensor* seqLenTensor, std::vector<std::string> WeightNames, std::vector<std::string> BiasNames, bool isBi);

    float * values {nullptr};
};

#endif
