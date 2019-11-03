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


#include "BatchStream.h"

BatchStream::BatchStream(int batchSize, std::string calibDir, std::string inputPrefix, DataType type) : 
          mBatchSize(batchSize)
        , mCalibDir(calibDir)
        , mInputPrefix(inputPrefix)
        , mDataType(type)
        , mTensorsAreAllocated(false)
    {

    }

//!
//! @brief Calculate the size in bytes for one batch
//!
int BatchStream::getTensorSize(const std::string& inputFileName){
        std::fstream fs;
        fs.open(inputFileName, std::fstream::in);
        if (!fs.good())
        {
            std::cout << inputFileName << ": Not Found!!!" << std::endl;
        }
        std::string str;
        std::stringstream ss(str);
        std::string dim;
        
        int tensorSize = 0;
        while(std::getline(fs, str, ','))
        {
            sanitizeValue(str);


            if (! str.empty()){
                ++tensorSize;
            }

        }
        
        return tensorSize;
}

void BatchStream::allocateTensors(const std::string& inputFileName){

    mBlobSize = getTensorSize(inputFileName);

    // Reserve space on host
    mBatch.resize(mBlobSize, 0);

    // Reserve space on device
    mDeviceInput = samplesCommon::safeCudaMalloc(mBlobSize * sizeof(float));

    mTensorsAreAllocated = true;
}


void BatchStream::fillBatchOnHost(const std::string& inputFileName)
{
    if(! tensorsAreAllocated()){
        allocateTensors(inputFileName);
    }

    std::fstream fs;
    std::string str;
    fs.open(inputFileName, std::fstream::in);
    if (!fs.good())
    {
        std::cout << inputFileName << ": Not Found!!!" << std::endl;
    }

    // read file into batch
    size_t inputCount = 0;
    while(std::getline(fs, str, ','))
    {
        
        sanitizeValue(str);

        if (str.empty()) break;

        if (mDataType == DataType::kINT32)
        {
            *(reinterpret_cast<int*>(&mBatch[inputCount++])) = std::stoi(str);
        }
        else
        {
            mBatch[inputCount++] = std::stof(str);
        }

    }
    assert(mBatch.size() == inputCount);
    fs.close();
}

