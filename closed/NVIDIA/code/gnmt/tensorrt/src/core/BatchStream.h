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

#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include <vector>
#include <set>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"
#include "utils.h"

//!
//! @class BatchStream
//! @brief Functionality to load tensors from a raw text file
//! into a batch on the host and on the GPU.
//!

std::string locateFile(const std::string& input);

class BatchStream
{
public:
    BatchStream(int batchSize, std::string calibDir, std::string inputPrefix, DataType type);

    ~BatchStream(){
        if (mDeviceInput != nullptr){
            CHECK_CUDA(cudaFree(mDeviceInput))  ;
        }  
    }

    int getBatchSize() const { return mBatchSize; }

    void * getBatchOnDevice(std::string suffix){
        std::string inputFileName = locateFile(getInputFileName(suffix));
        if(! tensorsAreAllocated()){
            allocateTensors(inputFileName);
        }

        fillBatchOnHost(inputFileName);
        CHECK_CUDA(cudaMemcpy(mDeviceInput, &mBatch[0], mBlobSize * sizeof(float), cudaMemcpyHostToDevice));
        return mDeviceInput;
    }

    std::string getPrefix() const{
        return mInputPrefix;
    }

    std::string getCalibDir() const{
        return mCalibDir;
    }

private:
    bool tensorsAreAllocated() const{
        return mTensorsAreAllocated;
    }

    void fillBatchOnHost(const std::string& inputFileName);

    void sanitizeValue(std::string& str){
        str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
        str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    }

    std::string getInputFileName(std::string suffix){
        return std::string(mCalibDir + "/" + mInputPrefix + suffix);
    }

    void allocateTensors(const std::string& inputFileName);

    int getTensorSize(const std::string& inputFileName);

    int mBatchSize{0};
    int mBlobSize{0};
    std::string mCalibDir;
    std::string mInputPrefix;
    std::vector<float> mBatch;
    void * mDeviceInput{nullptr};
    DataType mDataType {DataType::kFLOAT};
    bool mTensorsAreAllocated;
};

#endif
