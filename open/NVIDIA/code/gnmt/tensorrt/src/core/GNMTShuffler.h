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

#ifndef GNMT_SHUFFLER_H
#define GNMT_SHUFFLER_H

#include "GNMTEngine.h"
#include "params.h"

using namespace nvinfer1;

//!
//! The Shuffler takes two inputs:
//! - parentBeamIndices: A vector indicating which parent beam each new beam comes from.
//! - All the decoder hidden/cell states and attentions.
//! and shuffles them so that the outputs:
//! - The shuffled decoder hidden/cell states and attentions.
//! according to the parentBeamIndices.
//! For example, if parentBeamIndices is [2, 1, 1] and the input states are {(state_of_beam0),(state_of_beam1),(state_of_beam2)},
//! then the output states will be {(state_of_beam2),(state_of_beam1),(state_of_beam1)}.
//!
class Shuffler : public GNMTEngine
{
public:
    //!
    //! \post Initializes mDecoderStateInputNames (through initBufferNames)
    //! \post Initializes mDecoderStateOutputNames (through initBufferNames)
    //! \post Initializes mParentBeamIndicesName (through initBufferNames)
    //!
    Shuffler();

    Shuffler(std::shared_ptr<Config> config, bool profile);

    void setParentBeamIndicesInput(const void * data);

    void setAttentionInput(const void * data);

    void setHiddInput(int layerId, const void * data);

    void setCellInput(int layerId, const void * data);

    std::shared_ptr<CudaBufferRaw> getAttentionShuffledOutput() const;

    std::shared_ptr<CudaBufferRaw> getHiddShuffledOutput(int layerId) const;

    std::shared_ptr<CudaBufferRaw> getCellShuffledOutput(int layerId) const;

private:
    //!
    //! \brief Initialize the buffer names.
    //!
    //! \post Initializes mDecoderStateInputNames
    //! \post Initializes mDecoderStateOutputNames
    //! \post Initializes mParentBeamIndicesName
    //!
    //!
    void initBufferNames();
    
    string mParentBeamIndicesName;           //!< Buffer name for parentBeamIndices.
    vector<string> mDecoderStateInputNames;  //!< Buffer names for input decoder states (including attention).
    vector<string> mDecoderStateOutputNames; //!< Buffer names for output decoder states (including attention).

    //!
    //! \brief Doing nothing since Shuffler does not have weights.
    //!
    void importWeights(){};

    //!
    //! \brief Configures the network to do beam shuffling.
    //!
    void configureNetwork(nvinfer1::INetworkDefinition* network, int encoderMaxSeqLen);

    //!
    //! \brief Helper function to add gather layers for all the decoder states which will be shuffled.
    //!
    //! \param network The network.
    //! \param parentBeamIndices Tensor containing the indices of the previous beam for each new beam.
    //! \param decoderStateTensors Tensors containing attention and decoder states.
    //!
    //! \return A vector containing all the outputs for the gather layers.
    //!
    const vector<ITensor*> addGatherLayers(nvinfer1::INetworkDefinition* network, ITensor* parentBeamIndices, const vector<ITensor*>& decoderStateTensors);

    //!
    //! \brief Add an input layer to the network for an input decoder state.
    //!
    //! \param network The network.
    //! \param name The name for this input.
    //! \param dataType The data type of this input.
    //! \param dims The dimension of this input.
    //!
    //! \return The tensor of the input layer.
    //!
    ITensor* addDecoderStateInput(nvinfer1::INetworkDefinition* network, const string& name, const DataType dataType, const Dims dims);
};

#endif
