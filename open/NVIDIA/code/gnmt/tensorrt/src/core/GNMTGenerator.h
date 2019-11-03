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

#ifndef GNMT_GENERATOR_H
#define GNMT_GENERATOR_H

#include "GNMTEngine.h"

#include "params.h"

#include "BatchStream.h"
#include "Int8Calibrator.h"

using namespace nvinfer1;

//!
//! \brief Generates query to input the decoder using encoder and attention ouput.
//!

class Generator : public GNMTEngine
{
public:
    Generator();

    Generator(std::shared_ptr<Config> config, bool profile);

    void setEmdeddingIndicesInput(const void * data);

    void setAttentionInput(const void * data);

    void setEncoderLSTMOutputInput(const void * data);

    void setEncoderKeyTransformInput(const void * data);

    void setInputSequenceLengthsInput(const void * data);

    void setParentLogProbsInput(const void * data);

    void setLengthPenaltyInput(const void * data);

    void setHiddInput(int layerId, const void * data);

    void setCellInput(int layerId, const void * data);

    std::shared_ptr<CudaBufferRaw> getAttentionOutput() const;

    std::shared_ptr<CudaBufferRaw> getLogProbsCombinedOutput() const;

    std::shared_ptr<CudaBufferRaw> getBeamIndicesOutput() const;

    std::shared_ptr<CudaBufferRaw> getHiddOutput(int layerId) const;

    std::shared_ptr<CudaBufferRaw> getCellOutput(int layerId) const;

private:
    //!
    //! \brief Imports weights specific to the embedder and query.
    //!
    void importWeights();


    ITensor* addInput(nvinfer1::INetworkDefinition* network, const char* name, DataType type, Dims dimensions){
        auto tensor = network->addInput(name, type, dimensions);
        mConfig->addCalibrationDumpTensor(network, &tensor, tensor->getName());
        return tensor;
    }

    //!
    //! \brief Configures the network to add the query embedded lookup and the query LSTM layers.
    //!
    void configureNetwork(INetworkDefinition* network, int encoderMaxSeqLen);

    void configureInt8Mode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network) override;

    void configureCalibrationCacheMode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network) override;

    void addQuery(
        INetworkDefinition* network,
        ITensor* indexTensor,
        ITensor* attnTensor,
        ITensor* hiddTensor,
        ITensor* cellTensor,
        ITensor** queryOutput,
        ITensor** hiddL0Out,
        ITensor** cellL0Out);

    void importQueryWeights();

    //!
    //! \brief Adds the query embedded lookup portion of GNMT to the network.
    //!
    //! \note This is being implemented using a const layer to contain the lookup table
    //!       and a gather layer to perform the lookup in parallel for multiple token indices.
    //!
    ITensor* addQueryEmbedLookup(nvinfer1::INetworkDefinition* network, ITensor* input);

    //!
    //! \brief Adds the query LSTM portion of GNMT to the network.
    //!
    //! \note This is being implemented using using the RNNv2 and Elementwise layers.
    //!           1) The RNNv2 layers is used to represent both the bidirectional and unidirectional RNN layers.
    //!           2) The Elementwise layer is used to add previous RNN outputs to produce the residual input required
    //!              by the GNMT network.
    //!
    ILayer* addQuery(nvinfer1::INetworkDefinition* network, ITensor* embeddOut, ITensor* oldAttnOut, ITensor* cellStates, ITensor* hiddenStates);

    void addAttention(
        INetworkDefinition* network,
        ITensor* encOut,
        ITensor* encKeys,
        ITensor* decOut,
        ITensor* inputSeqLengths,
        ITensor** attentionOutput,
        int encoderMaxSeqLen);

    void importAttentionWeights();

    //!
    //! \brief Adds the attention layer layer with its input tensors
    //!
    ILayer* addAttention(
        nvinfer1::INetworkDefinition* network,
        ITensor* encOut,
        ITensor* encKeys,
        ITensor* decOut,
        ITensor* mask,
        int encoderMaxSeqLen);

    void addDecoder(
        INetworkDefinition* network,
        ITensor* decInput,
        ITensor* attnOut,
        const std::vector<ITensor*>& mHiddStates,
        const std::vector<ITensor*>& mCellStates,
        std::vector<ITensor*>& mHiddStatesOut,
        std::vector<ITensor*>& mCellStatesOut,
        ITensor** decOut);

    void addDecoderPlugin(
        nvinfer1::INetworkDefinition* network, 
        ITensor* decInput, 
        ITensor* attnOut, 
        const std::vector<ITensor*> &mHiddStates, 
        const std::vector<ITensor*> &mCellStates,
        std::vector<ITensor*>& mHiddStatesOut,
        std::vector<ITensor*>& mCellStatesOut,
        ITensor** decOut,
        DataType prec);

    void importDecoderWeights();

    void addScorer(
        INetworkDefinition* network,
        ITensor* parentLogProbs,
        ITensor* decoderOutput,
        ITensor* lengthPenaly,
        ITensor** newCombinedLikelihoodsTensor,
        ITensor** newRayOptionIndicesTensor);

    void importScorerWeights();

    //!
    //! \brief Resize weights of size inputRows*inputCols to weights of size ouputRows*outputCols
    //!
    void resizeScorerWeights(
        const std::string name,
        const unsigned int inputRowws,
        const unsigned int inputCols,
        const unsigned int outputRows,
        const unsigned int outputCols);

    //!
    //! \brief Populate tranposed convolution weights and dummy bias weights.
    //!
    std::pair<Weights, Weights> getConvolutionWeightsAndBias();

    static int32_t getPaddedVocSize(const Config& config);

    std::unique_ptr<Int8MinMaxCalibrator> mCalibrator;
};

#endif
