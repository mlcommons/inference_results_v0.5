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

#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_H

#include <iostream>
#include "NvInfer.h"
#include "BatchStream.h"
#include <iterator>
#include <string>
#include "BatchSequencer.h"
#include <unordered_map>
using TensorScales = std::unordered_map<std::string, float>;

//! \class Int8CalibratorImpl
//!
//! \brief Implements common functionality for Int8 calibrators.
//!
//! \param readCache The calibrator can be ran in two modes, depending on the value of readCache.
//! if readCache is false, we will build a new cache, based on BatchStream data.
//! if readCache is true, we will use an existing cache.
//!
class Int8CalibratorImpl
{
public:
    Int8CalibratorImpl(std::map<std::string, std::shared_ptr<BatchStream>>& streams, int firstBatch, std::string networkName, int batchSize, bool readCache, std::string calibCacheFName)
        : mStreams (streams)
        , mNetworkName (networkName)
        , mBatchSize(batchSize)
        , mReadCache (readCache)
        , mCalibCacheFName (calibCacheFName)
    {
        if(! mReadCache){
            mBatchSequencer.reset(new BatchSequencer(streams));
            std::cout << "Calibrating on " << mBatchSequencer->getNumSequences() << " sequences." << std::endl;
        }
    }

    virtual ~Int8CalibratorImpl()
    {
    }

    int getBatchSize() const { return mBatchSize; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (mBatchSequencer == nullptr || ! mBatchSequencer->hasNextSuffix() || mReadCache){
            return false;
        }

        std::string suffix = mBatchSequencer->getNextSuffix();

        for (int i = 0; i < nbBindings; ++i){
            auto batchStreamIt = mStreams.find(std::string(names[i]));

            if (batchStreamIt == mStreams.end()){
                std::cout << names[i] << " not found." << std::endl;
                assert(batchStreamIt != mStreams.end());
            }

            bindings[i] = batchStreamIt->second->getBatchOnDevice(suffix); 
        }
        
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        if(! mReadCache){
            return nullptr;
        }

        mCalibrationCache.clear();
        std::ifstream input(mCalibCacheFName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length)
    {
        std::cout << "Writing to " << mCalibCacheFName << std::endl;
        std::ofstream output(mCalibCacheFName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }
private:
    std::map<std::string, std::shared_ptr<BatchStream>> mStreams; 

    std::string mNetworkName;
    int mBatchSize {0};
    bool mReadCache{true};
    std::vector<char> mCalibrationCache;
    const std::string mCalibCacheFName;
    std::unique_ptr<BatchSequencer> mBatchSequencer {nullptr};

};

//! \class Int8MinMaxCalibrator
//!
//! \brief Implements Min Max calibrator.
//!  CalibrationAlgoType is kMINMAX_CALIBRATION.
//!
class Int8MinMaxCalibrator : public IInt8MinMaxCalibrator
{
public:
    Int8MinMaxCalibrator(std::map<std::string, shared_ptr<BatchStream>> streams, int firstBatch, std::string networkName, int batchSize, bool readCache, std::string calibCacheFName)
            : mImpl(streams, firstBatch, networkName, batchSize, readCache, calibCacheFName)
    {
    }

    int getBatchSize() const override { return mImpl.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

    // Get the calibrator's scales for each tensor
    bool getScales(TensorScales& tensorScales)
    {
        size_t size;
        const char* cache = static_cast<const char*>(readCalibrationCache(size));
        if (!cache)
        {
            return false;
        }

        std::istringstream stream(std::string(cache, size));
        stream >> std::hex;
        std::string calibAlgo;

        stream >> calibAlgo;
        
        // remove the space at the end of the line
        stream.get();
        char buffer[1000];
        while (stream.getline(buffer, 1000))
        {
            char* colon = strchr(buffer, ':');
            *colon = 0;
            int raw_data = strtol(colon + 2, nullptr, 16);
            float scale;
            static_assert(sizeof(raw_data) == sizeof(scale), "data type size doesn't match");
            memcpy(&scale, &raw_data, sizeof(raw_data));
            tensorScales[buffer] = scale;
        }
        return true;
    }

private:
    Int8CalibratorImpl mImpl;
};

#endif // INT8_CALIBRATOR_H
