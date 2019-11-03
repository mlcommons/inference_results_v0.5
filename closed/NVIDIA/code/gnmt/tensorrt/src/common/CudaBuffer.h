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

// TODO: Replace with include guard.
#pragma once
#include "cuda_runtime_api.h"
#include <fstream>
#include <iomanip>

template <typename T>
class CudaBuffer 
{
public:
        //! 
        //! \brief Allocate a buffer of size nbElems * sizeof(T) on Cuda enabled device.
        //! 
        CudaBuffer(size_t nbElems = 0) : mNbElems(nbElems) 
        {
            if (nbElems)
            {
                cudaMalloc(reinterpret_cast<void**>(&mData), nbElems * sizeof(T));
                if (mData == nullptr)
                    throw std::runtime_error("CudaMalloc Failed.");
            } 
        }

        CudaBuffer(const std::vector<T>& other) : mNbElems(other.size()) 
        {
            if (!other.empty())
            {
                cudaMalloc(reinterpret_cast<void**>(&mData), mNbElems * sizeof(T));
                if (mData == nullptr)
                    throw std::runtime_error("CudaMalloc Failed.");
                cudaMemcpy(reinterpret_cast<void*>(mData), other.data(), mNbElems * sizeof(T), cudaMemcpyHostToDevice);   
            } 
        }

        //! 
        //! \brief Move ownership of buffer to new buffer.
        //! 
        CudaBuffer(CudaBuffer&& other) noexcept
        {
           moveCudaBuffer(std::forward<CudaBuffer&&>(other));
        }

        //! 
        //! \brief We don't want to make a deep copy of the CudaBuffer. 
        //!        We also don't want a shallow copy as that leads to dangling pointers. 
        //! 
        CudaBuffer(const CudaBuffer& other) = delete;


        //! 
        //! \brief Initialize CudaBuffer using the data stored in the vector. 
        //! 
        template <typename ElemType>
        CudaBuffer& operator=(const std::vector<ElemType>& other)
        {
            size_t size = other.size() * sizeof(ElemType);
            assert(size == mNbElems * sizeof(T));
            cudaMemcpy(reinterpret_cast<void*>(mData), other.data(), size, cudaMemcpyHostToDevice);   
            return *this;
        }

        //! 
        //! \brief Free the underlaying data when the CudaBuffer goes out of scope. 
        //! 
        ~CudaBuffer() 
        {
            cudaFree(mData);
        }

        //! 
        //! \brief Move ownership of buffer to new buffer.
        //! 
        CudaBuffer& operator=(CudaBuffer&& other) noexcept
        {
            moveCudaBuffer(std::forward<CudaBuffer&&>(other));
            return *this;
        }

        //! 
        //! \brief Get pointer to data.
        //! 
        T* data()
        {
            return mData;
        }

        //! 
        //! \brief Get const pointer to data.
        //! 
        const T* data() const
        {
            return mData;
        }

        //! 
        //! \brief Get size of buffer as the number of elements.
        //! 
        size_t size() const
        {
            return mNbElems;
        }

        void fillWithZero()
        {
            if (mNbElems)
            {
                cudaMemset(mData, 0, mNbElems * sizeof(T));
            }
        }

private:
        //! 
        //! \brief Helper function use to move ownership from one CudaBuffer to the other.
        //! 
        void moveCudaBuffer(CudaBuffer&& other)
        {
            cudaFree(mData);
            mData = other.mData;
            mNbElems = other.mNbElems;
            other.mData = nullptr;
            other.mNbElems = 0;
        }

        T* mData = nullptr;
        size_t mNbElems = 0;
};

using CudaBufferRaw = CudaBuffer<uint8_t>;
using CudaBufferInt8 = CudaBuffer<int8_t>;
using CudaBufferInt32 = CudaBuffer<int32_t>;
using CudaBufferFP16 = CudaBuffer<uint16_t>;
using CudaBufferFP32 = CudaBuffer<float>;

template <typename T>
class HostBuffer 
{
public:
        //! 
        //! \brief Allocate a buffer of size nbElems * sizeof(T) on Cuda enabled device.
        //! 
        HostBuffer(size_t nbElems = 0) : mNbElems(nbElems) 
        {
            if (nbElems) 
            {
                cudaMallocHost(reinterpret_cast<void**>(&mData), nbElems * sizeof(T));
                if (mData == nullptr)
                    throw std::runtime_error("CudaMallocHost Failed.");
            } 
        }

        //! 
        //! \brief Move ownership of buffer to new buffer.
        //! 
        HostBuffer(HostBuffer&& other) noexcept
        {
           moveHostBuffer(std::forward<HostBuffer&&>(other));
        }

        //! 
        //! \brief We don't want to make a deep copy of the CudaBuffer. 
        //!        We also don't want a shallow copy as that leads to dangling pointers. 
        //! 
        HostBuffer(const HostBuffer& other) = delete;


        //! 
        //! \brief Initialize CudaBuffer using the data stored in the vector. 
        //! 
        template <typename ElemType>
        HostBuffer& operator=(const std::vector<ElemType>& other)
        {
            size_t size = other.size() * sizeof(ElemType);
            assert(size == mNbElems * sizeof(T));
            std::copy((const uint8_t *)(&other[0]), ((const uint8_t *)(&other[0])) + size, (uint8_t *)mData);
            return *this;
        }

        //! 
        //! \brief Free the underlaying data when the CudaBuffer goes out of scope. 
        //! 
        ~HostBuffer() 
        {
            cudaFreeHost(mData);
        }

        //! 
        //! \brief Move ownership of buffer to new buffer.
        //! 
        HostBuffer& operator=(HostBuffer&& other) noexcept
        {
            moveHostBuffer(std::forward<HostBuffer&&>(other));
            return *this;
        }

        //! 
        //! \brief Get pointer to data.
        //! 
        T* data()
        {
            return mData;
        }

        //! 
        //! \brief Get const pointer to data.
        //! 
        const T* data() const
        {
            return mData;
        }

        //! 
        //! \brief Get size of buffer as the number of elements.
        //! 
        size_t size() const
        {
            return mNbElems;
        }

private:
        //! 
        //! \brief Helper function use to move ownership from one CudaBuffer to the other.
        //! 
        void moveHostBuffer(HostBuffer&& other)
        {
            cudaFreeHost(mData);
            mData = other.mData;
            mNbElems = other.mNbElems;
            other.mData = nullptr;
            other.mNbElems = 0;
        }

        T* mData = nullptr;
        size_t mNbElems = 0;
};

using HostBufferRaw = HostBuffer<uint8_t>;
using HostBufferInt32 = HostBuffer<int32_t>;
using HostBufferFP32 = HostBuffer<float>;

//! 
//! \brief Dump the CudaBuffer into the stream.
//! 
template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const CudaBuffer<T>& other) 
{
    T* hostPtr = new T[other.size()];
    cudaMemcpy(hostPtr, other.data(), other.size()*sizeof(T), cudaMemcpyDeviceToHost);
    for(size_t j = 0; j < other.size(); j++)
    {
        stream << std::setprecision(3) << hostPtr[j]  << ", ";
        if(j % 10 == 9)
            stream << std::endl;    
    }
    delete hostPtr;

    return stream;
}
