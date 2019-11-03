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

#ifndef TF_WEIGHTS_MANAGER_H
#define TF_WEIGHTS_MANAGER_H

#include "NvInfer.h"
#include "NvUtils.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include "common.h"
#include "half.h"

enum class TFCellType : int {
    kBASIC_RNN_CELL = 0,
    kBASIC_LSTM_CELL = 1,
    kCUDNN_COMPATIBLE_LSTM_CELL = 2,
    kCUDNN_COMPATIBLE_GRU_CELL = 3
};

enum class TFWeightsType : int {
    kWEIGHT = 0,
    kBIAS = 1
};

struct PersistentLSTMPluginInfo
{
    int inputSize;
    int hiddenSize;
    int layerCount;
    nvinfer1::RNNOperation op;
    bool isBi;

    PersistentLSTMPluginInfo(int inputSize, int hiddenSize, int layerCount, bool isBi)
    : inputSize(inputSize), hiddenSize(hiddenSize), layerCount(layerCount), isBi(isBi)
    {
        op = RNNOperation::kLSTM;
    }
};

class TFWeightsManager
{
public:
    //!
    //! \brief Default constructor does nothing.
    //!
    inline TFWeightsManager() { };

    //!
    //! \brief Loads specified weights from the dump file.
    //!
    //! \param filePath The path to the dump file.
    //! \param weightNames The name of the weights to be imported. If no vector is provided, then all the weights in the file will be loaded.
    //!
    //! \note This constructor simply calls the importTFWeights function.
    //!
    //! \see importTFWeights()
    //!
    inline TFWeightsManager(const std::string filePath, std::vector<std::string> weightNames = {});

    //!
    //! \brief Cleans up resources allocated by the TFWeightsManager.
    //!
    inline ~TFWeightsManager();

    //!
    //! \brief Loads specified weights from the dump file.
    //!
    //! \param filePath The path to the dump file.
    //! \param weightNames The name of the weights to be imported. If no vector is provided, then all the weights in the file will be loaded.
    //!
    inline void importTFWeights(const std::string filePath, std::vector<std::string> weightNames = {});

    //!
    //! \brief Give a list of weightNames in order, this function converts those weights from TensorFlow
    //!        to TensorRT format and then sets the converted weights for a given RNN layer.
    //!
    //! \param rnn The RNN layer for which the user wishes to set the weights.
    //! \param weightNames The names of the weights supplied in order. In order means that the
    //!                    weight names are provided in the vector ordered by layer index and then by input and recurrent.
    //!                    To visualize this, think of this vector as a flattened matrix with shape {LAYER_COUNT, 2}.
    //! \param cellType The type of TensorFLow cell being used.
    //! \param weightType specifies the type of weight the tensor is storing.
    //!
    inline void setTFRNNv2(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string> weightNames, const TFCellType cellType, const TFWeightsType weightType);

    inline void setTFPersistentLSTM(std::vector<std::string> weightNames, const TFCellType cellType, const TFWeightsType weightType, nvinfer1::Weights& weight, PersistentLSTMPluginInfo info);
    //!
    //! \brief provides access to the underlying weightsMap.
    //!
    inline std::map<std::string, nvinfer1::Weights> getWeightsMap();

    inline void clear();

private:
    //!
    //! \brief returns a vector containing the TensorFlow gate order for a given RNNOperation.
    //!
    //! \param op The RNN operation for which the user wishes to get the gateOrder.
    //!
    inline std::vector<nvinfer1::RNNGateType> getGateOrder(nvinfer1::RNNOperation op);

    //!
    //! \brief returns the number of gates that need to be set for a given RNNOperation.
    //!
    //! \param op The RNN operation for which the user wishes to get the gateCount.
    //!
    inline int getGateCount(nvinfer1::RNNOperation op);

    //!
    //! \brief returns the outer most dimension of the weights buffer. This differs based on TFCellType.
    //!
    //! \param cellType The type of TensorFLow cell being used.
    //!
    inline int getSubMatrixCount(TFCellType cellType);

    //!
    //! \brief Converts a given weights object from TensorFlow to TensorRT for the 0th sub-layer of the RNN layer.
    //!
    //! \param input The weights object being converted to TensorRT format.
    //! \param rnn The RNN layer for which the weights are being converted. Used to extract hyperparameters.
    //! \param cellType The type of TensorFLow cell being used.
    //!
    //! \note This special function exists for the 0th layer because it has non-uniform dimension.
    //!
    inline void convertRNNv2WeightsLayer0(nvinfer1::Weights& input, nvinfer1::IRNNv2Layer * rnn, TFCellType cellType);

    inline void convertPersistentLSTMWeightsLayer0(nvinfer1::Weights& input, PersistentLSTMPluginInfo info, TFCellType cellType);

    //!
    //! \brief Converts a given weights object from TensorFlow to TensorRT format.
    //!
    //! \param input The weights object being converted to TensorRT format.
    //! \param rnn The RNN layer for which the weights are being converted. Used to extract hyperparameters.
    //! \param cellType The type of TensorFLow cell being used.
    //!
    inline void convertRNNv2WeightsGeneral(nvinfer1::Weights& input, nvinfer1::IRNNv2Layer * rnn, TFCellType cellType);

    inline void convertPersistentLSTMWeightsGeneral(nvinfer1::Weights& input, PersistentLSTMPluginInfo info, TFCellType cellType);

    //!
    //! \brief Converts a set of weights from TensorFlow to TensorRT format.
    //!
    //! \param rnn The RNN layer for which the weights are being converted. Used to extract hyperparameters.
    //! \param weightNames The name of the weights in order of layer index that you want to convert.
    //! \param cellType The type of TensorFLow cell being used.
    //!
    inline void convertTFRNNv2Weights(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType);

    inline void convertPersistentLSTMWeights(PersistentLSTMPluginInfo info, std::vector<std::string>& weightNames, const TFCellType cellType);

    //!
    //! \brief Sets the weights for a RNNv2 layer.
    //!
    //! \param rnn The RNN layer for which the weights are being set.
    //! \param weightNames The name of the weights in order of layer index that you want to set.
    //! \param cellType The type of TensorFLow cell being used.
    //!
    inline void setRNNv2Weights(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType);

    //!
    //! \brief Converts and sets the weights for a RNNv2 layer.
    //!
    //! \param rnn The RNN layer for which the weights are being set.
    //! \param weightNames The name of the weights in order of layer index that you want to set.
    //! \param cellType The type of TensorFLow cell being used.
    //!
    inline void setTFRNNv2Weights(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType);

    inline void setPersistentLSTMWeights(PersistentLSTMPluginInfo info, std::vector<std::string>& weightNames, const TFCellType cellType, nvinfer1::Weights& weight);

    //!
    //! \brief Convers and sets the biases for a RNNv2 layer.
    //!
    //! \param rnn The RNN layer for which the biases are being set.
    //! \param weightNames The name of the biases in order of layer index that you want to set.
    //! \param cellType The type of TensorFLow cell being used.
    //!
    inline void setTFRNNv2Bias(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType, float forgetBias);
    inline void setPersistentLSTMBias(PersistentLSTMPluginInfo info, std::vector<std::string>& weightNames, const TFCellType cellType, nvinfer1::Weights& weight, float forgetBias);

    inline void copyFloat2Half(char * halfArray, const char * floatArray, size_t size);

    std::map<std::string, nvinfer1::Weights> weightsMap;
};

inline TFWeightsManager::TFWeightsManager(const std::string filePath, std::vector<std::string> weightNames)
{
    importTFWeights(filePath, weightNames);
}

inline TFWeightsManager::~TFWeightsManager()
{
    // Clean up resources
    for (auto& mem : weightsMap)
        free((void*)(mem.second.values));

    weightsMap.clear();
}

inline void TFWeightsManager::importTFWeights(const std::string filePath, std::vector<std::string> weightNames)
{
    // open dump file
    std::ifstream input(filePath);
    if(!input.is_open())
    {
        string error = "Invalid file path." + filePath + " not found.";
        throw std::invalid_argument(error);
    }

    // get number of tensors in file
    int32_t count;
    input >> count;
    if(count <= 0)
    {
        string error = "Tensor Count: " + std::to_string(count) + ". Tensorflow dump file might be corrupted.";
        throw std::runtime_error(error);
    }

    auto it = std::unique(weightNames.begin(), weightNames.end());
    if (it != weightNames.end())
    {
        string error = "weightNames vector contains duplicates.";
        throw std::invalid_argument(error);
    }

    // convert weightNames vector to unordered_set for internal usage
    std::unordered_set<std::string> setOfNames(weightNames.begin(), weightNames.end());

    // iterate over all of the tensors and extract the requested ones
    while(count--)
    {
        if (setOfNames.empty()) break;

        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // parse name and nvinfer1::DataType
        std::string name;
        uint32_t type;
        input >> name >> std::dec >> type;
        wt.type = static_cast<nvinfer1::DataType>(type);

        // extract shape
        std::string temp, shape;
        std::getline(std::getline(input, temp, '('), shape, ')');

        // calculate count based on shape
        wt.count = 1;
        std::istringstream shapeStream(shape);
        while(std::getline(shapeStream, temp, ',')) wt.count *= std::stoul(temp);
        size_t numOfBytes = samplesCommon::getElementSize(wt.type) * wt.count;

        // skip reading of weights if name is not in the set of names requested for extraction
        if (setOfNames.find(name) == setOfNames.end())
        {
            input.seekg(input.tellg() + static_cast<std::streamoff>(2 + numOfBytes));
            continue;
        }
        else
        {
            setOfNames.erase(name);
        }

        // Read weight values
        input.seekg(input.tellg() + static_cast<std::streamoff>(1)); // skip space char
        char* wtVals = static_cast<char*>(malloc(numOfBytes));
        input.read(wtVals, numOfBytes);
        input.seekg(input.tellg() + static_cast<std::streamoff>(1)); // skip new-line char
        wt.values = wtVals;

        weightsMap[name] = wt;
    }

    if(!setOfNames.empty())
    {
        string error = "Weight names not found.\n\n";
        error += "Following weight names are not present in Tensorflow dump file:\n";
        for (auto name : setOfNames)
            error += name + "\n";
        throw std::runtime_error(error);
    }

    input.close();
}

inline void TFWeightsManager::setTFRNNv2(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string> weightNames, const TFCellType cellType, const TFWeightsType weightType)
{
    // Check validity of rnn layer
    if (rnn == nullptr)
    {
        string error = "RNN layer pointer is null.";
        throw std::invalid_argument(error);
    }

    // Check for duplication of weight names
    auto it = std::unique(weightNames.begin(), weightNames.end());
    if (it != weightNames.end())
    {
        string error = "weightNames vector contains duplicates.";
        throw std::invalid_argument(error);
    }

    // check if all weightNames exist in the weightsMap
    std::vector<std::string> missingNames;
    for (auto name : weightNames)
    {
        if(weightsMap.find(name) == weightsMap.end())
            missingNames.push_back(name);
    }
    if (!missingNames.empty())
    {
        string error = "A set of weight names was not found in the weights map.\n\n";
        error += "Following weight names were not found:\n";
        for (auto name : missingNames)
            error += name + "\n";
        throw std::invalid_argument(error);
    }


    // Check for see if sizes match
    const size_t layerCountScale = (rnn->getDirection() == nvinfer1::RNNDirection::kBIDIRECTION)? 2 : 1;
    const size_t layerCount = layerCountScale * rnn->getLayerCount();
    if (getSubMatrixCount(cellType) == 2 && layerCount != weightNames.size())
    {
        string error = "weightNames vector contains an invalid number of names.\n\n";
        error += "Expected Count: " + std::to_string(layerCount) + "\n";
        error += "Observed Count: " + std::to_string(weightNames.size())+ "\n";
        throw std::invalid_argument(error);
    }
    else if (getSubMatrixCount(cellType) == 1 && 2*layerCount != weightNames.size())
    {
        string error = "weightNames vector contains an invalid number of names.\n\n";
        error += "Expected Count: " + std::to_string(2*layerCount) + "\n";
        error += "Observed Count: " + std::to_string(weightNames.size()) + "\n";
        throw std::invalid_argument(error);
    }

    // Set weights
    if (weightType == TFWeightsType::kWEIGHT)
        setTFRNNv2Weights(rnn, weightNames, cellType);
    else
        setTFRNNv2Bias(rnn, weightNames, cellType, 1.0f);
}

inline std::map<std::string, nvinfer1::Weights> TFWeightsManager::getWeightsMap()
{
    return weightsMap;
}

inline void TFWeightsManager::clear()
{
    weightsMap.clear();
}

inline std::vector<nvinfer1::RNNGateType> TFWeightsManager::getGateOrder(nvinfer1::RNNOperation op)
{
    switch (op)
    {
        case nvinfer1::RNNOperation::kRELU:
        case nvinfer1::RNNOperation::kTANH:
            return std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kINPUT});
        case nvinfer1::RNNOperation::kLSTM:
            return std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kINPUT,
                                                       nvinfer1::RNNGateType::kCELL,
                                                       nvinfer1::RNNGateType::kFORGET,
                                                       nvinfer1::RNNGateType::kOUTPUT});
        case nvinfer1::RNNOperation::kGRU:
            return std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kRESET,
                                                       nvinfer1::RNNGateType::kUPDATE});
        default:
            assert(0 && "Invalid RNNOperation.");
    }
}

inline int TFWeightsManager::getGateCount(nvinfer1::RNNOperation op)
{
    switch (op)
    {
        case nvinfer1::RNNOperation::kRELU:
        case nvinfer1::RNNOperation::kTANH:
            return 1;
        case nvinfer1::RNNOperation::kLSTM:
            return 4;
        case nvinfer1::RNNOperation::kGRU:
            return 2;
        default:
            assert(0 && "Invalid RNNOperation.");
    }
}

inline int TFWeightsManager::getSubMatrixCount(TFCellType cellType)
{
    switch (cellType)
    {
        case TFCellType::kBASIC_RNN_CELL:
        case TFCellType::kBASIC_LSTM_CELL:
            return 2;
        case TFCellType::kCUDNN_COMPATIBLE_LSTM_CELL:
        case TFCellType::kCUDNN_COMPATIBLE_GRU_CELL:
            return 1;
        default:
            assert(0 && "Invalid TFCellType.");
    }

    return -1;
}

inline void TFWeightsManager::convertRNNv2WeightsLayer0(nvinfer1::Weights& input, nvinfer1::IRNNv2Layer * rnn, TFCellType cellType)
{
    // Get layer hyperparameters
    const int gateCount = getGateCount(rnn->getOperation());
    const int dataSize = rnn->getDataLength();
    const int hiddenSize = rnn->getHiddenSize();
    const int weightsCount = gateCount * hiddenSize * (dataSize + hiddenSize);

    if (input.count != weightsCount)
    {
        string error = "Count of weights is not consistent with the expected count based on RNN hyperparameters.\n\n";
        error += "Expected Count: " + std::to_string(weightsCount) + "\n";
        error += "Observed Count: " + std::to_string(input.count) + "\n";
        throw std::invalid_argument(error);
    }

    void* data= malloc(samplesCommon::getElementSize(input.type) * input.count);

    // Convert input weights
    nvinfer1::Weights inputWeights{.type = input.type, .values = input.values, .count = gateCount*dataSize*hiddenSize};
    int inWeightsDims[3]{dataSize, gateCount, hiddenSize};
    int inWeightsOrder[3]{ 2, 0, 1};
    nvinfer1::utils::reshapeWeights(inputWeights, inWeightsDims, inWeightsOrder, data, 3);
    nvinfer1::utils::transposeSubBuffers(data, input.type, 1, dataSize * hiddenSize, gateCount);

    // Convert the recurrent weights if there are any
    if (getSubMatrixCount(cellType) > 1)
    {
        void* recurValuesIn = (void *)((char *)input.values + gateCount*dataSize*hiddenSize*samplesCommon::getElementSize(input.type));
        void* recurValuesOut = (void *)((char *)data + gateCount*dataSize*hiddenSize*samplesCommon::getElementSize(input.type));
        nvinfer1::Weights recurWeights{.type = input.type, .values = recurValuesIn, .count = gateCount*hiddenSize*hiddenSize};
        int recurWeightsDims[3]{hiddenSize, gateCount, hiddenSize};
        int recurWeightsOrder[3]{ 2, 0, 1};
        nvinfer1::utils::reshapeWeights(recurWeights, recurWeightsDims, recurWeightsOrder, recurValuesOut, 3);
        nvinfer1::utils::transposeSubBuffers(recurValuesOut, input.type, 1, hiddenSize * hiddenSize, gateCount);
    }

    // free old values and store converted ones
    free((void *)input.values);
    input.values = data;
}

inline void TFWeightsManager::convertRNNv2WeightsGeneral(nvinfer1::Weights& input, nvinfer1::IRNNv2Layer * rnn, TFCellType cellType)
{
    // Get layer hyperparameters
    const int subMatrixCount = getSubMatrixCount(cellType);
    const int gateCount = getGateCount(rnn->getOperation());
    const int hiddenSize = rnn->getHiddenSize();
    const int weightsCount = 2 * gateCount * hiddenSize * hiddenSize;

    if (input.count != weightsCount)
    {
        string error = "Count of weights is not consistent with the expected count based on RNN hyperparameters.\n\n";
        error += "Expected Count: " + std::to_string(weightsCount) + "\n";
        error += "Observed Count: " + std::to_string(input.count) + "\n";
        throw std::invalid_argument(error);
    }

    void* data= malloc(samplesCommon::getElementSize(input.type) * input.count);

    // Convert the underlying weights
    int dims[4]{subMatrixCount, hiddenSize, gateCount, hiddenSize};
    int order[4]{ 0, 3, 1, 2};
    nvinfer1::utils::reshapeWeights(input, dims, order, data, 4);
    nvinfer1::utils::transposeSubBuffers(data, input.type, 2, hiddenSize * hiddenSize, 4);

    free((void *)input.values);
    input.values = data;
}

inline void TFWeightsManager::convertTFRNNv2Weights(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType)
{
    bool isBidirectional = (rnn->getDirection() == nvinfer1::RNNDirection::kBIDIRECTION);

    size_t idx = 0;

    // Convert Layer 0 weights
    convertRNNv2WeightsLayer0(weightsMap[weightNames[idx]], rnn, cellType);
    if (isBidirectional)
    {
        // if cell type is platform independent (Basic) then, then backward input weights for layer 0 are at index 1.
        // Otherwise, they are at index 2.
        idx = (getSubMatrixCount(cellType) == 2)? 1 : 2;
        if (idx == 2)
        {
            // if cell type is cudnn compatible, then the forward recurrent weights for layer 0 need to be handled separately.
            convertRNNv2WeightsGeneral(weightsMap[weightNames[idx - 1]], rnn, cellType);
        }

        // convert backward weights for layer 0
        convertRNNv2WeightsLayer0(weightsMap[weightNames[idx]], rnn, cellType);
    }

    // Convert all other layers weights
    for (size_t i = idx + 1, j = weightNames.size(); i < j; i++)
    {
        convertRNNv2WeightsGeneral(weightsMap[weightNames[i]], rnn, cellType);
    }
}

inline void TFWeightsManager::setRNNv2Weights(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType)
{
    // Get layer hyperparameters
    const size_t dataSize = rnn->getDataLength();
    const size_t hiddenSize = rnn->getHiddenSize();
    const std::vector<nvinfer1::RNNGateType> gateOrder = getGateOrder(rnn->getOperation());
    const size_t gateCount = gateOrder.size();
    const nvinfer1::DataType type = weightsMap[weightNames[0]].type;
    const size_t sizeOfElement = samplesCommon::getElementSize(type);
    const bool isBidirectional = (rnn->getDirection() == nvinfer1::RNNDirection::kBIDIRECTION);

    // set weights for sub-layer 0
    size_t inWtsOffset = 0, recurWtsOffset = 0;
    const char* inWtsPtrL0FW = static_cast<const char*>(weightsMap[weightNames[0]].values);
    const char* recurWtsPtrL0FW = (getSubMatrixCount(cellType) == 2)? inWtsPtrL0FW + gateCount*dataSize*hiddenSize*sizeOfElement
                                        : static_cast<const char*>(weightsMap[weightNames[1]].values);
    const char* inWtsPtrL0BW = (!isBidirectional)? NULL : (getSubMatrixCount(cellType) == 2)? static_cast<const char*>(weightsMap[weightNames[1]].values)
                                        : static_cast<const char*>(weightsMap[weightNames[2]].values);
    const char* recurWtsPtrL0BW = (!isBidirectional)? NULL : (getSubMatrixCount(cellType) == 2)? inWtsPtrL0BW + gateCount*dataSize*hiddenSize*sizeOfElement
                                        : static_cast<const char*>(weightsMap[weightNames[3]].values);
    for (size_t j = 0; j < gateCount; j++)
    {
        nvinfer1::Weights inWeightsFW{.type = type,
            .values = static_cast<const void *>(inWtsPtrL0FW + inWtsOffset*sizeOfElement),
            .count = static_cast<int64_t>(dataSize*hiddenSize)};
        nvinfer1::Weights recurWeightsFW{.type = type,
            .values = static_cast<const void *>(recurWtsPtrL0FW + recurWtsOffset*sizeOfElement),
            .count = static_cast<int64_t>(hiddenSize*hiddenSize)};

        rnn->setWeightsForGate(0, gateOrder[j], true, inWeightsFW);
        rnn->setWeightsForGate(0, gateOrder[j], false, recurWeightsFW);

        if (isBidirectional)
        {
            nvinfer1::Weights inWeightsBW{.type = type,
                .values = static_cast<const void *>(inWtsPtrL0BW + inWtsOffset*sizeOfElement),
                .count = static_cast<int64_t>(dataSize*hiddenSize)};
            nvinfer1::Weights recurWeightsBW{.type = type,
                .values = static_cast<const void *>(recurWtsPtrL0BW + recurWtsOffset*sizeOfElement),
                .count = static_cast<int64_t>(hiddenSize*hiddenSize)};

            rnn->setWeightsForGate(1, gateOrder[j], true, inWeightsBW);
            rnn->setWeightsForGate(1, gateOrder[j], false, recurWeightsBW);
        }

        inWtsOffset += dataSize*hiddenSize;
        recurWtsOffset += hiddenSize*hiddenSize;
    }

    // set weights for all other sub-layers
    const size_t indexScale = (getSubMatrixCount(cellType) == 2)? 1 : 2;
    const size_t layerCountScale = (isBidirectional)? 2 : 1;
    const size_t layerCount = layerCountScale * rnn->getLayerCount();
    const size_t startIdx =  layerCountScale;
    for (size_t i = startIdx; i < layerCount; i++)
    {
        size_t offset = 0;

        const char* inWtsPtr = static_cast<const char*>(weightsMap[weightNames[i*indexScale]].values);
        const char* recurWtsPtr = (getSubMatrixCount(cellType) == 2)? inWtsPtr + gateCount*hiddenSize*hiddenSize*sizeOfElement :
                                        static_cast<const char*>(weightsMap[weightNames[i*indexScale + 1]].values);
        for (size_t j = 0; j < gateCount; j++)
        {
            nvinfer1::Weights inWeights{.type = type,
                .values = static_cast<const void *>(inWtsPtr + offset*sizeOfElement),
                .count = static_cast<int64_t>(hiddenSize*hiddenSize)};
            nvinfer1::Weights recurWeights{.type = type,
                .values = static_cast<const void *>(recurWtsPtr + offset*sizeOfElement),
                .count = static_cast<int64_t>(hiddenSize*hiddenSize)};

            rnn->setWeightsForGate(i, gateOrder[j], true, inWeights);
            rnn->setWeightsForGate(i, gateOrder[j], false, recurWeights);

            offset += hiddenSize*hiddenSize;
        }
    }
}

inline void TFWeightsManager::setTFRNNv2Weights(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType)
{
    // convert the weights in place from TensorFlow's to TensorRT's format
    convertTFRNNv2Weights(rnn, weightNames, cellType);

    // set weights for given layer
    setRNNv2Weights(rnn, weightNames, cellType);
}



inline void TFWeightsManager::setTFRNNv2Bias(nvinfer1::IRNNv2Layer * rnn, std::vector<std::string>& weightNames, const TFCellType cellType, float forgetBias)
{
    // Get layer hyperparameters
    const size_t hiddenSize = rnn->getHiddenSize();
    const std::vector<nvinfer1::RNNGateType> gateOrder = getGateOrder(rnn->getOperation());
    const size_t gateCount = gateOrder.size();
    const nvinfer1::DataType type = weightsMap[weightNames[0]].type;
    const size_t sizeOfElement = samplesCommon::getElementSize(type);
    const bool isBidirectional = (rnn->getDirection() == nvinfer1::RNNDirection::kBIDIRECTION);

    // Create zeroed out weights for all recurrent biases
    char* zeroBiasPtr = static_cast<char*>(malloc(sizeOfElement*gateCount*hiddenSize));
    std::fill(zeroBiasPtr, zeroBiasPtr + sizeOfElement*gateCount*hiddenSize, 0);

    // Set bias for all layers
    const size_t layerCountScale = (isBidirectional)? 2 : 1;
    const size_t layerCount = layerCountScale * rnn->getLayerCount();
    const size_t indexScale = (getSubMatrixCount(cellType) == 2)? 1 : 2;
    for (size_t i = 0; i < layerCount; i++)
    {
        size_t offset = 0;
        const char* inBiasPtr = static_cast<const char*>(weightsMap[weightNames[i*indexScale]].values);
        const char* recurBiasPtr = (getSubMatrixCount(cellType) == 2)? zeroBiasPtr :
                                        static_cast<const char*>(weightsMap[weightNames[i*indexScale + 1]].values);
        for (size_t j = 0; j < gateCount; j++)
        {
            nvinfer1::Weights inBias{.type = type,
                .values = static_cast<const void *>(inBiasPtr + offset*sizeOfElement),
                .count = static_cast<int64_t>(hiddenSize)};
            nvinfer1::Weights recurBias{.type = type,
                .values = static_cast<const void *>(recurBiasPtr + offset*sizeOfElement),
                .count = static_cast<int64_t>(hiddenSize)};

            if (gateOrder[j] == nvinfer1::RNNGateType::kFORGET){
                assert(sizeOfElement == sizeof(float)); // Only handle FP32 for now.
                float* tempRecurBias = (float*)(zeroBiasPtr + offset*sizeOfElement);
                for (int k = 0; k < recurBias.count; k++){
                    tempRecurBias[k] = forgetBias;
                }
            }

            rnn->setBiasForGate(i, gateOrder[j], true, inBias);
            rnn->setBiasForGate(i, gateOrder[j], false, recurBias);

            offset += hiddenSize;
        }
    }
}

// Weight initialization pass for Persistent LSTM Plugin
inline void TFWeightsManager::setTFPersistentLSTM(std::vector<std::string> weightNames, const TFCellType cellType, const TFWeightsType weightType, nvinfer1::Weights& weight, PersistentLSTMPluginInfo info)
{
    // Check for duplication of weight names
    auto it = std::unique(weightNames.begin(), weightNames.end());
    if (it != weightNames.end())
    {
        string error = "weightNames vector contains duplicates.";
        throw std::invalid_argument(error);
    }

    // check if all weightNames exist in the weightsMap
    std::vector<std::string> missingNames;
    for (auto name : weightNames)
    {
        if(weightsMap.find(name) == weightsMap.end())
            missingNames.push_back(name);
    }
    if (!missingNames.empty())
    {
        string error = "A set of weight names was not found in the weights map.\n\n";
        error += "Following weight names were not found:\n";
        for (auto name : missingNames)
            error += name + "\n";
        throw std::invalid_argument(error);
    }

    // only support LSTM
    if ((cellType != TFCellType::kBASIC_LSTM_CELL) && (cellType != TFCellType::kCUDNN_COMPATIBLE_LSTM_CELL))
    {
        string error = "Invalid Cell Type for Persistent LSTM PLugin";
        throw std::invalid_argument(error);
    }


    // Check for see if sizes match
    const size_t layerCountScale = info.isBi ? 2 : 1;
    const size_t layerCount = layerCountScale * info.layerCount;
    if (getSubMatrixCount(cellType) == 2 && layerCount != weightNames.size())
    {
        string error = "weightNames vector contains an invalid number of names.\n\n";
        error += "Expected Count: " + std::to_string(layerCount) + "\n";
        error += "Observed Count: " + std::to_string(weightNames.size())+ "\n";
        throw std::invalid_argument(error);
    }
    else if (getSubMatrixCount(cellType) == 1 && 2*layerCount != weightNames.size())
    {
        string error = "weightNames vector contains an invalid number of names.\n\n";
        error += "Expected Count: " + std::to_string(2*layerCount) + "\n";
        error += "Observed Count: " + std::to_string(weightNames.size()) + "\n";
        throw std::invalid_argument(error);
    }

    // Set weights
    if (weightType == TFWeightsType::kWEIGHT)
    {
        convertPersistentLSTMWeights(info, weightNames, cellType);
        setPersistentLSTMWeights(info, weightNames, cellType, weight);
    }
    else
    {
        setPersistentLSTMBias(info, weightNames, cellType, weight, 1.0f);
    }
}

inline void TFWeightsManager::convertPersistentLSTMWeightsLayer0(nvinfer1::Weights& input, PersistentLSTMPluginInfo info, TFCellType cellType)
{
    // Get layer hyperparameters
    const int gateCount = getGateCount(info.op);
    const int dataSize = info.inputSize;
    const int hiddenSize = info.hiddenSize;
    const int weightsCount = gateCount * hiddenSize * (dataSize + hiddenSize);

    if (input.count != weightsCount)
    {
        string error = "convertPersistentLSTMWeightsLayer0: Count of weights is not consistent with the expected count based on RNN hyperparameters.\n\n";
        error += "Expected Count: " + std::to_string(weightsCount) + "\n";
        error += "Observed Count: " + std::to_string(input.count) + "\n";
        throw std::invalid_argument(error);
    }

    void* data= malloc(samplesCommon::getElementSize(input.type) * input.count);

    // Convert input weights
    nvinfer1::Weights inputWeights{.type = input.type, .values = input.values, .count = gateCount*dataSize*hiddenSize};
    int inWeightsDims[3]{dataSize, gateCount, hiddenSize};
    int inWeightsOrder[3]{ 2, 0, 1};
    nvinfer1::utils::reshapeWeights(inputWeights, inWeightsDims, inWeightsOrder, data, 3);
    nvinfer1::utils::transposeSubBuffers(data, input.type, 1, dataSize * hiddenSize, gateCount);

    // Convert the recurrent weights if there are any
    if (getSubMatrixCount(cellType) > 1)
    {
        void* recurValuesIn = (void *)((char *)input.values + gateCount*dataSize*hiddenSize*samplesCommon::getElementSize(input.type));
        void* recurValuesOut = (void *)((char *)data + gateCount*dataSize*hiddenSize*samplesCommon::getElementSize(input.type));
        nvinfer1::Weights recurWeights{.type = input.type, .values = recurValuesIn, .count = gateCount*hiddenSize*hiddenSize};
        int recurWeightsDims[3]{hiddenSize, gateCount, hiddenSize};
        int recurWeightsOrder[3]{ 2, 0, 1};
        nvinfer1::utils::reshapeWeights(recurWeights, recurWeightsDims, recurWeightsOrder, recurValuesOut, 3);
        nvinfer1::utils::transposeSubBuffers(recurValuesOut, input.type, 1, hiddenSize * hiddenSize, gateCount);
    }

    // free old values and store converted ones
    free((void *)input.values);
    input.values = data;
}

inline void TFWeightsManager::convertPersistentLSTMWeightsGeneral(nvinfer1::Weights& input, PersistentLSTMPluginInfo info, TFCellType cellType)
{
    // Get layer hyperparameters
    const int subMatrixCount = getSubMatrixCount(cellType);
    const int gateCount = getGateCount(info.op);
    const int hiddenSize = info.hiddenSize;
    const int weightsCount = 2 * gateCount * hiddenSize * hiddenSize;

    if (input.count != weightsCount)
    {
        string error = "convertPersistentLSTMWeightsGeneral: Count of weights is not consistent with the expected count based on RNN hyperparameters.\n\n";
        error += "Expected Count: " + std::to_string(weightsCount) + "\n";
        error += "Observed Count: " + std::to_string(input.count) + "\n";
        throw std::invalid_argument(error);
    }

    void* data= malloc(samplesCommon::getElementSize(input.type) * input.count);

    // Convert the underlying weights
    int dims[4]{subMatrixCount, hiddenSize, gateCount, hiddenSize};
    int order[4]{ 0, 3, 1, 2};
    nvinfer1::utils::reshapeWeights(input, dims, order, data, 4);
    nvinfer1::utils::transposeSubBuffers(data, input.type, 2, hiddenSize * hiddenSize, 4);

    free((void *)input.values);
    input.values = data;
}

inline void TFWeightsManager::convertPersistentLSTMWeights(PersistentLSTMPluginInfo info, std::vector<std::string>& weightNames, const TFCellType cellType)
{
    bool isBidirectional = info.isBi;

    size_t idx = 0;

    // Convert Layer 0 weights
    convertPersistentLSTMWeightsLayer0(weightsMap[weightNames[idx]], info, cellType);
    if (isBidirectional)
    {
        // if cell type is platform independent (Basic) then, then backward input weights for layer 0 are at index 1.
        // Otherwise, they are at index 2.
        idx = (getSubMatrixCount(cellType) == 2)? 1 : 2;
        if (idx == 2)
        {
            // if cell type is cudnn compatible, then the forward recurrent weights for layer 0 need to be handled separately.
            convertPersistentLSTMWeightsGeneral(weightsMap[weightNames[idx - 1]], info, cellType);
        }

        // convert backward weights for layer 0
        convertPersistentLSTMWeightsLayer0(weightsMap[weightNames[idx]], info, cellType);
    }

    // Convert all other layers weights
    for (size_t i = idx + 1, j = weightNames.size(); i < j; i++)
    {
        convertPersistentLSTMWeightsGeneral(weightsMap[weightNames[i]], info, cellType);
    }
}

inline void TFWeightsManager::setPersistentLSTMWeights(PersistentLSTMPluginInfo info, std::vector<std::string>& weightNames, const TFCellType cellType, nvinfer1::Weights& weight)
{

    // Get layer hyperparameters
    const size_t dataSize = info.inputSize;
    const size_t hiddenSize = info.hiddenSize;
    const std::vector<nvinfer1::RNNGateType> gateOrder = getGateOrder(info.op);
    const size_t gateCount = gateOrder.size();

    const nvinfer1::DataType type = weightsMap[weightNames[0]].type;
    const nvinfer1::DataType convertType = weight.type;

    const size_t convertSizeOfElement = samplesCommon::getElementSize(convertType);
    const size_t sizeOfElement = samplesCommon::getElementSize(type);

    const int bidirectionFactor = info.isBi ? 2 : 1;

    if (convertType != nvinfer1::DataType::kHALF)
    {
        string error = "Persistent LSTM PLugin only support half precision";
        throw std::invalid_argument(error);
    }


    const bool isBidirectional = info.isBi;

    if ((info.layerCount != 1) && (isBidirectional))
    {
        string error = "Persistent LSTM PLugin does not support multiLayer bi-directional lstm, please seperate the first layer out";
        throw std::invalid_argument(error);
    }

    // set weights for sub-layer 0
    size_t inWtsOffset = 0, recurWtsOffset = 0, recurCombinedWtsOffset = 0, inCombinedWtsOffset = 0;
    const size_t totalRecurWtsOffset = gateCount * bidirectionFactor * hiddenSize * hiddenSize * info.layerCount * convertSizeOfElement;

    const char* inWtsPtrL0FW = static_cast<const char*>(weightsMap[weightNames[0]].values);
    const char* recurWtsPtrL0FW = (getSubMatrixCount(cellType) == 2)? inWtsPtrL0FW + gateCount*dataSize*hiddenSize*sizeOfElement
                                        : static_cast<const char*>(weightsMap[weightNames[1]].values);
    const char* inWtsPtrL0BW = (!isBidirectional)? NULL : (getSubMatrixCount(cellType) == 2)? static_cast<const char*>(weightsMap[weightNames[1]].values)
                                        : static_cast<const char*>(weightsMap[weightNames[2]].values);
    const char* recurWtsPtrL0BW = (!isBidirectional)? NULL : (getSubMatrixCount(cellType) == 2)? inWtsPtrL0BW + gateCount*dataSize*hiddenSize*sizeOfElement
                                        : static_cast<const char*>(weightsMap[weightNames[3]].values);


    // I am going to create a new entry in weightMap so that TFWeightsManager can delete the memory for me
    // I don't want to touch any existing memory stored here (not sure if it will be used elsewhere)
    std::string newWeightname = weightNames[0] + std::string("_combined");
    char* weightMemoryAllocated = new char[weight.count * convertSizeOfElement];

    for (size_t j = 0; j < gateCount; j++)
    {

        copyFloat2Half(weightMemoryAllocated + totalRecurWtsOffset + inCombinedWtsOffset, inWtsPtrL0FW + inWtsOffset, dataSize*hiddenSize);
        inCombinedWtsOffset += dataSize*hiddenSize*convertSizeOfElement;

        copyFloat2Half(weightMemoryAllocated + recurCombinedWtsOffset, recurWtsPtrL0FW + recurWtsOffset, hiddenSize*hiddenSize);
        recurCombinedWtsOffset += hiddenSize*hiddenSize*convertSizeOfElement;

        inWtsOffset += dataSize*hiddenSize*sizeOfElement;
        recurWtsOffset += hiddenSize*hiddenSize*sizeOfElement;
    }

    inWtsOffset = 0;
    recurWtsOffset = 0;

    if (isBidirectional)
    {
        for (size_t j = 0; j < gateCount; j++)
        {
            copyFloat2Half(weightMemoryAllocated + totalRecurWtsOffset + inCombinedWtsOffset, inWtsPtrL0BW + inWtsOffset, dataSize*hiddenSize);
            inCombinedWtsOffset += dataSize*hiddenSize*convertSizeOfElement;

            copyFloat2Half(weightMemoryAllocated + recurCombinedWtsOffset, recurWtsPtrL0BW + recurWtsOffset, hiddenSize*hiddenSize);
            recurCombinedWtsOffset += hiddenSize*hiddenSize*convertSizeOfElement;

            inWtsOffset += dataSize*hiddenSize*sizeOfElement;
            recurWtsOffset += hiddenSize*hiddenSize*sizeOfElement;
        }

    }


    // set weights for all other sub-layers
    const size_t indexScale = (getSubMatrixCount(cellType) == 2)? 1 : 2;
    const size_t layerCountScale = (isBidirectional)? 2 : 1;
    const size_t layerCount = layerCountScale * info.layerCount;
    const size_t startIdx =  layerCountScale;

    for (size_t i = startIdx; i < layerCount; i++)
    {
        size_t offset = 0;

        const char* inWtsPtr = static_cast<const char*>(weightsMap[weightNames[i*indexScale]].values);
        const char* recurWtsPtr = (getSubMatrixCount(cellType) == 2)? inWtsPtr + gateCount*hiddenSize*hiddenSize*sizeOfElement :
                                        static_cast<const char*>(weightsMap[weightNames[i*indexScale + 1]].values);

        for (size_t j = 0; j < gateCount; j++)
        {

            copyFloat2Half(weightMemoryAllocated + totalRecurWtsOffset + inCombinedWtsOffset, inWtsPtr + offset, hiddenSize*hiddenSize);
            inCombinedWtsOffset += hiddenSize*hiddenSize*convertSizeOfElement;

            copyFloat2Half(weightMemoryAllocated + recurCombinedWtsOffset, recurWtsPtr + offset, hiddenSize*hiddenSize);
            recurCombinedWtsOffset += hiddenSize*hiddenSize*convertSizeOfElement;

            offset += hiddenSize*hiddenSize*sizeOfElement;
        }
    }
    weight.values = static_cast<const void *>(weightMemoryAllocated);
    weightsMap[newWeightname] = weight;
}

inline void TFWeightsManager::setPersistentLSTMBias(PersistentLSTMPluginInfo info, std::vector<std::string>& weightNames, const TFCellType cellType, nvinfer1::Weights& weight, float forgetBias)
{
    // Get layer hyperparameters
    const size_t hiddenSize = info.hiddenSize;
    const std::vector<nvinfer1::RNNGateType> gateOrder = getGateOrder(info.op);
    const size_t gateCount = gateOrder.size();
    const nvinfer1::DataType type = weightsMap[weightNames[0]].type;
    const nvinfer1::DataType convertType = weight.type;

    const size_t convertSizeOfElement = samplesCommon::getElementSize(convertType);
    const size_t sizeOfElement = samplesCommon::getElementSize(type);
    const bool isBidirectional = info.isBi;

    if (convertType != nvinfer1::DataType::kHALF)
    {
        string error = "Persistent LSTM PLugin only support half precision";
        throw std::invalid_argument(error);
    }

    if ((info.layerCount != 1) && (isBidirectional))
    {
        string error = "Persistent LSTM PLugin does not support multiLayer bi-directional lstm, please seperate the first layer out";
        throw std::invalid_argument(error);
    }


    std::string newWeightname = weightNames[0] + std::string("_combined");
    char* weightMemoryAllocated = new char[weight.count * convertSizeOfElement];

    // Create zeroed out weights for all recurrent biases
    char* zeroBiasPtr = static_cast<char*>(malloc(sizeOfElement*gateCount*hiddenSize));
    std::fill(zeroBiasPtr, zeroBiasPtr + sizeOfElement*gateCount*hiddenSize, 0);

    // Set bias for all layers
    const size_t layerCountScale = (isBidirectional)? 2 : 1;
    const size_t layerCount = layerCountScale * info.layerCount;
    const size_t indexScale = (getSubMatrixCount(cellType) == 2)? 1 : 2;
    size_t combinedWtsOffset = 0;

    for (size_t i = 0; i < layerCount; i++)
    {
        size_t offset = 0;
        const char* inBiasPtr = static_cast<const char*>(weightsMap[weightNames[i*indexScale]].values);

        for (size_t j = 0; j < gateCount; j++)
        {
            if (gateOrder[j] == nvinfer1::RNNGateType::kFORGET){
                assert(sizeOfElement == sizeof(float)); // Only handle FP32 for now.
                float* tempRecurBias = (float*)(zeroBiasPtr + offset);
                for (size_t k = 0; k < hiddenSize; k++){
                    tempRecurBias[k] = forgetBias;
                }
            }

            copyFloat2Half(weightMemoryAllocated + combinedWtsOffset, inBiasPtr + offset, hiddenSize);
            combinedWtsOffset += hiddenSize * convertSizeOfElement;
            offset += hiddenSize * sizeOfElement;
        }

        offset = 0;
        const char* recurBiasPtr = (getSubMatrixCount(cellType) == 2)? zeroBiasPtr :
                                        static_cast<const char*>(weightsMap[weightNames[i*indexScale + 1]].values);
        for (size_t j = 0; j < gateCount; j++)
        {
            if (gateOrder[j] == nvinfer1::RNNGateType::kFORGET){
                assert(sizeOfElement == sizeof(float)); // Only handle FP32 for now.
                float* tempRecurBias = (float*)(zeroBiasPtr + offset);
                for (size_t k = 0; k < hiddenSize; k++){
                    tempRecurBias[k] = forgetBias;
                }
            }

            copyFloat2Half(weightMemoryAllocated + combinedWtsOffset, recurBiasPtr + offset, hiddenSize);
            combinedWtsOffset += hiddenSize * convertSizeOfElement;
            offset += hiddenSize * sizeOfElement;
        }

    }

    weight.values = static_cast<const void *>(weightMemoryAllocated);
    weightsMap[newWeightname] = weight;
}


inline void TFWeightsManager::copyFloat2Half(char * halfArray, const char * floatArray, size_t size)
{
    half_float::half * halfPtr = (half_float::half *)(halfArray);
    const float * floatPtr = (const float *)(floatArray);
    for (size_t i = 0; i < size ; i++)
    {
        halfPtr[i] = (half_float::half)floatPtr[i];
    }
}

#endif // TF_WEIGHTS_MANAGER_H
