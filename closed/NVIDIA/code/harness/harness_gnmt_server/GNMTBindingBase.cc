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

#include "GNMTBindingBase.h"
std::map<std::string, void *> GNMTBindingBase::getEngineBoundCudaBuffers(const std::unordered_set<std::string>& tensorNames) const
{
    std::map<std::string, void *> outputBuffers;

    for (auto name : tensorNames)
    {
        outputBuffers[name] = getEngineBoundCudaBuffer(name);
    }

    return outputBuffers;
}

void* GNMTBindingBase::getEngineBoundCudaBuffer(const std::string & tensorName) const
{

    void* outputBuffer{nullptr};
    std::string error = "";

    const int bufferIdx = mModel->BindingId(tensorName);
    if (bufferIdx == -1)
    {
        error += tensorName + "\n";
    }
    else
    {
        outputBuffer = mBindings->DeviceAddress(bufferIdx);
    }

    if (!error.empty()) throw std::runtime_error(errorHeader + error);

    return outputBuffer;
}


void GNMTBindingBase::initInputCudaBufferToZeroes(const std::string& tensorName, DataType dtype, size_t nbElements)
{
    if (dtype == DataType::kFLOAT){
        initInputCudaBufferToZeroes<float>(tensorName, nbElements);
    }
    else if (dtype == DataType::kHALF){
        initInputCudaBufferToZeroes<uint16_t>(tensorName, nbElements);
    }
    else if (dtype == DataType::kINT32){
        initInputCudaBufferToZeroes<int32_t>(tensorName, nbElements);
    }
    else{
        assert(false);
    }
}

void GNMTBindingBase::setInputCudaBuffer(const std::string& tensorName, void * ptr)
{
    const int bufferIdx = mModel->BindingId(tensorName);
    auto binding_size = mModel->GetBinding(bufferIdx).bytesPerBatchItem * mBatchSize;

    // TO DO: I used CudaDeviceMemory because inference_manager use that
    mBindings->SetDeviceAddress(bufferIdx, std::move(std::make_unique<BufferStackDescriptor<CudaDeviceMemory>>(
            ptr, binding_size)));
}

void GNMTBindingBase::setInputCudaBufferCopy(const std::string& tensorName, void * ptr)
{
    const int bufferIdx = mModel->BindingId(tensorName);
    auto dst = mBindings->DeviceAddress(bufferIdx);
    auto binding_size = mModel->GetBinding(bufferIdx).bytesPerBatchItem * mBatchSize;

    CHECK_EQ(cudaMemcpyAsync(dst, ptr, binding_size, cudaMemcpyDeviceToDevice, stream()), CUDA_SUCCESS);
}
