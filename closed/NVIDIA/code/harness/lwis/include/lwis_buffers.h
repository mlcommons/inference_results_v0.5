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

#ifndef LWIS_BUFFERS_H
#define LWIS_BUFFERS_H

#include "NvInfer.h"
#include "half.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <new>
#include <random>
#include <fstream>

namespace lwis {

/* read in engine file into character array */
inline size_t GetModelStream(std::vector<char> &dst, std::string engineName) {
  size_t size{0};
  std::ifstream file(engineName, std::ios::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    dst.resize(size);
    file.read(dst.data(), size);
    file.close();
  }

  return size;
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
  case nvinfer1::DataType::kINT32: return 4;
  case nvinfer1::DataType::kFLOAT: return 4;
  case nvinfer1::DataType::kHALF: return 2;
  case nvinfer1::DataType::kINT8: return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    switch(format)
    {
        case nvinfer1::TensorFormat::kCHW2: spv = 2; break;
        case nvinfer1::TensorFormat::kCHW4: spv = 4; break;
        case nvinfer1::TensorFormat::kHWC8: spv = 8; break;
        case nvinfer1::TensorFormat::kCHW16: spv = 16; break;
        case nvinfer1::TensorFormat::kCHW32: spv = 32; break;
        case nvinfer1::TensorFormat::kLINEAR:
        default: spv = 1; break;
    }
    if (spv > 1)
    {
        assert(d.nbDims >= 3); // Vectorized format only makes sense when nbDims>=3.
        d_new.d[d_new.nbDims - 3] = roundUp(d_new.d[d_new.nbDims - 3], spv);
    }
    return std::accumulate(d_new.d, d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer()
        : mByteSize(0)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size)
        : mByteSize(size)
    {
        if (!allocFn(&mBuffer, mByteSize))
            throw std::bad_alloc();
    }

    GenericBuffer(GenericBuffer&& buf)
        : mByteSize(buf.mByteSize)
        , mBuffer(buf.mBuffer)
    {
        buf.mByteSize = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mByteSize = buf.mByteSize;
            mBuffer = buf.mBuffer;
            buf.mByteSize = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data() { return mBuffer; }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const { return mBuffer; }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t size() const { return mByteSize; }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mByteSize;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
      return cudaMallocHost(ptr, size) == cudaSuccess;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const { 
        cudaFreeHost(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    //!
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int& batchSize)
        : mEngine(engine)
        , mBatchSize(batchSize)
    {
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            // Create host and device buffers
            size_t vol = lwis::volume(mEngine->getBindingDimensions(i), mEngine->getBindingFormat(i));
            size_t elementSize = lwis::getElementSize(mEngine->getBindingDataType(i));
            size_t allocationSize = static_cast<size_t>(mBatchSize) * vol * elementSize;
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(allocationSize);
            manBuf->hostBuffer = HostBuffer(allocationSize);
            //std::cout << "Allocated Address: " << manBuf->hostBuffer.data() << std::endl;
            mHostBindings.emplace_back(manBuf->hostBuffer.data());
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::move(manBuf));
            //std::cout << (mEngine->bindingIsInput(i) ? "Input" : "Output") << " BindingName: " << mEngine->getBindingName(i) << std::endl;
        }
    }

    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getDeviceBindings() { return mDeviceBindings; }

    //!
    //! \brief Returns a vector of device buffers.
    //!
    const std::vector<void*>& getDeviceBindings() const { return mDeviceBindings; }

    //!
    //! \brief Returns a vector of host buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getHostBindings() { return mHostBindings; }

    //!
    //! \brief Returns a vector of host buffers.
    //!
    const std::vector<void*>& getHostBindings() const { return mHostBindings; }

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getDeviceBuffer(const std::string& tensorName) const { return getBuffer(false, tensorName); }

    //!
    //! \brief Returns the device buffer corresponding to index.
    //!
    void* getDeviceBuffer(const size_t index) const { return getBuffer(false, index); }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getHostBuffer(const std::string& tensorName) const { return getBuffer(true, tensorName); }

    //!
    //! \brief Returns the host buffer corresponding to index.
    //!
    void* getHostBuffer(const size_t index) const { return getBuffer(true, index); }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[index]->hostBuffer.size();
    }

    //!
    //! \brief Returns the volume of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t volume(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return lwis::volume(mEngine->getBindingDimensions(index), mEngine->getBindingFormat(index));
    }

    //!
    //! \brief Returns the elementSize of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t elementSize(const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return kINVALID_SIZE_VALUE;
        return lwis::getElementSize(mEngine->getBindingDataType(index));
    }

    //!
    //! \brief Dump host buffer with specified tensorName to ostream.
    //!        Prints error message to std::ostream if no such tensor can be found.
    //!
    void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
        {
            os << "Invalid tensor name" << std::endl;
            return;
        }
        void* buf = mManagedBuffers[index]->hostBuffer.data();
        size_t bufSize = mManagedBuffers[index]->hostBuffer.size();
        nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
        size_t rowCount = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);

        os << "[" << mBatchSize;
        for (int i = 0; i < bufDims.nbDims; i++)
            os << ", " << bufDims.d[i];
        os << "]" << std::endl;
        switch (mEngine->getBindingDataType(index))
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break;
        case nvinfer1::DataType::kINT8: assert(0 && "Int8 network-level input and output is not supported"); break;
        }
    }

    //!
    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    //!
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    //!
    void copyInputToDevice(void *src = nullptr, size_t size = 0, size_t index = 0) { memcpyBuffers(true, false, false, 0, src, size, index); }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    //!
    void copyOutputToHost(void *dst = nullptr, size_t size = 0, size_t index = 0) { memcpyBuffers(false, true, false, 0, dst, size, index); }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0, void *src = nullptr, size_t size = 0, size_t index = 0) { memcpyBuffers(true, false, true, stream, src, size, index); }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(const cudaStream_t& stream = 0, void *dst = nullptr, size_t size = 0, size_t index = 0) { memcpyBuffers(false, true, true, stream, dst, size, index); }

    ~BufferManager() = default;

private:
    void* getBuffer(const bool isHost, const std::string& tensorName) const
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void* getBuffer(const bool isHost, const size_t index) const {
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0, void *buf = nullptr, size_t size = 0, size_t index = 0)
    {
        for (int i = 0; i < mEngine->getNbBindings(); i++)
        {
            void* dstPtr = deviceToHost ? (buf ? buf : mManagedBuffers[i]->hostBuffer.data()) : static_cast<char *>(mManagedBuffers[i]->deviceBuffer.data()) + (buf ? index * size : 0);
            const void* srcPtr = deviceToHost ? static_cast<char *>(mManagedBuffers[i]->deviceBuffer.data()) + (buf ? 0 : index * size) : (buf ? buf : mManagedBuffers[i]->hostBuffer.data());
            const size_t byteSize = buf && size ? size : mManagedBuffers[i]->hostBuffer.size();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
            {
                if (async)
                    cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
                else
                    cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    int mBatchSize;                                              //!< The batch size
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mHostBindings;                            //!< The vector of host buffers needed for engine execution
    std::vector<void*> mDeviceBindings;                          //!< The vector of device buffers needed for engine execution
};

} // namespace lwis

#endif // LWIS_BUFFERS_H
