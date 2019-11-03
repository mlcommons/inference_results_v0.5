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

#pragma once

#include <vector>
#include <string>

#include "tensorrt/laboratory/core/affinity.h"
#include "tensorrt/laboratory/core/thread_pool.h"
#include "tensorrt/laboratory/infer_bench.h"
#include "tensorrt/laboratory/infer_runner.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/bindings.h"
#include "tensorrt/laboratory/buffers.h"
#include "tensorrt/laboratory/model.h"
#include "tensorrt/laboratory/core/memory/memory_stack.h"
#include "tensorrt/laboratory/core/memory/descriptor.h"
#include "tensorrt/laboratory/cuda/memory/cuda_device.h"
#include "tensorrt/laboratory/inference_manager.h"
#include <iomanip>
#include <memory>
#include <unordered_set>
#include <sys/stat.h>   // Create directory to dump tensors
#include <sstream>
#include <ctime>        // embed date/time in debug directory

#include "params.h"

using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::Bindings;
using trtlab::TensorRT::Buffers;
using trtlab::TensorRT::Model;
using trtlab::Memory::MemoryStack;
using trtlab::Memory::Descriptor;
using trtlab::Memory::CudaDeviceMemory;
using trtlab::TensorRT::InferenceManager;
using trtlab::TensorRT::ExecutionContext;
using trtlab::TensorRT::SubExecutionContext;

class GNMTBindingBase{
public:
    GNMTBindingBase(std::shared_ptr<Model> model, std::shared_ptr<Config> config):
    mConfig(config), mModel{model}
    {}

    //!
    //! \brief Reset the Binding shared pointer to release resources
    //!
    void reset()
    {
        mBindings.reset();
    }

    //!
    //! \brief Synchronize bindings 
    //!
    //! \note Every binding has one associated cuda stream
    //!
    void bindingsSynchronize()
    {
        mBindings->Synchronize();
    }

    //!
    //! \brief Get the cuda stream associated with binding
    //!
    inline cudaStream_t stream() const { return mBindings->Stream(); };

    //!
    //! \brief return a shared pointer of the bindind
    //!
    std::shared_ptr<Bindings> getBindings() const { return mBindings; }

    //!
    //! \brief Create bindings from a given buffers
    //!
    //! \note buffers input has the memory and CreateBindings() merely sets up pointers
    //!
    void createBindings(std::shared_ptr<Buffers> buffers, size_t total_count)
    {
        mBindings = buffers->CreateBindings(mModel);
        mBindings -> SetBatchSize(total_count);
        mBatchSize = total_count;
    }

    //!
    //! \brief Set up input buffer by copying data from a host vector
    //!
    //!
    template <typename T>
    void setInputCudaBuffer(const std::string& tensorName, std::vector<T> input)
    {
        const int bufferIdx = mModel->BindingId(tensorName);
        mBindings->CopyToDevice(bufferIdx, input.data(), input.size() * sizeof(T));
    }

    //!
    //! \brief Set up input buffer by copying data from host
    //!
    //!
    void setInputCudaBufferFromHost(const std::string& tensorName, void * ptr, size_t size)
    {
        const int bufferIdx = mModel->BindingId(tensorName);
        mBindings->CopyToDevice(bufferIdx, ptr, size);
    }

    //!
    //! \brief Initialize input cuda buffers with zeroes
    //!
    //!
    void initInputCudaBufferToZeroes(const std::string& tensorName, DataType dtype, size_t nbElements);

    //!
    //! \brief Set up input buffer by pointing to device memory
    //!
    //!
    void setInputCudaBuffer(const std::string& tensorName, void * ptr);

    //!
    //! \brief Set up input buffer by copying data from device
    //!
    //!
    void setInputCudaBufferCopy(const std::string& tensorName, void * ptr);

    //!
    //! \brief Get the output buffers pointes based on vectors of output tensor names
    //!
    //!
    std::map<std::string, void *> getEngineBoundCudaBuffers(const std::unordered_set<std::string>& tensorNames) const;

    //!
    //! \brief Get the output buffer pointe based on output tensor name
    //!
    //!
    void* getEngineBoundCudaBuffer(const std::string & tensorName) const;

    //!
    //! \brief Set SubExecutionContext
    //!
    //!
    void setExecutionContext(std::shared_ptr<SubExecutionContext> ctx)
    {
        mCtx = ctx;
    }

    //!
    //! \brief Return SubExecutionContext
    //!
    //!
    auto getExecutionContext() -> std::shared_ptr<SubExecutionContext>
    {
        CHECK(mCtx) << "Call setExecutionContext() first";
        return mCtx;
    }

    //!
    //! \brief Get shared model pointer
    //!
    std::shared_ptr<Model> GetModelSmartPtr() const { return mModel; }
    
private:
    std::shared_ptr<Config> mConfig;
    const std::string mConfigFileName {"config.json"};

    int mBatchSize{0};

    std::shared_ptr<Model> mModel;
    std::shared_ptr<Bindings> mBindings;
    std::shared_ptr<SubExecutionContext> mCtx;

    // error messages
    std::string errorHeader = "Tensor name is neither an input nor an ouput of the network.\n\nThe following names were not found:\n";

    template<typename MemoryType>
    class BufferStackDescriptor final : public Descriptor<MemoryType>
    {
      public:
        BufferStackDescriptor(void* ptr, size_t size)
            : Descriptor<MemoryType>(ptr, size, MemoryType::Type() + "Desc")
        {
        }
        ~BufferStackDescriptor() final override {}
    };

    template <typename T>
    void initInputCudaBufferToZeroes(const std::string& tensorName, size_t nbElements){
        std::vector<T> vec (nbElements, T());
        setInputCudaBuffer(tensorName, vec);
    }
};


class EncoderBindingBase: public GNMTBindingBase
{
public:
    EncoderBindingBase(std::shared_ptr<Model> model, std::shared_ptr<Config> config):
    GNMTBindingBase(model, config)
    {

    }

    void setEmdeddingIndicesInput(void * data)
    {
        setInputCudaBuffer("Encoder_embedding_indices", data);
    }

    void setInputSequenceLengthsInput(void * data)
    {
        setInputCudaBuffer("Encoder Sequence Lengths", data);
    }

    void * getLSTMOutput() const
    {
        return getEngineBoundCudaBuffer("Encoder_lstm_output");
    }

    void * getKeyTransformOutput() const
    {
        return getEngineBoundCudaBuffer("Encoder_key_transform");
    }

    void * getHiddOutput(int layerId) const
    {
        if (layerId == 0)
            return getEngineBoundCudaBuffer("enc_l0_hid_BW");
        else
            return getEngineBoundCudaBuffer((std::string("enc_l") + std::to_string(layerId) + "_hid").c_str());
    }

    void * getCellOutput(int layerId) const
    {
        if (layerId == 0)
            return getEngineBoundCudaBuffer("enc_l0_cell_BW");
        else
            return getEngineBoundCudaBuffer((std::string("enc_l") + std::to_string(layerId) + "_cell").c_str());
    }

};

class GeneratorBindingBase: public GNMTBindingBase
{
public:
    GeneratorBindingBase(std::shared_ptr<Model> model, std::shared_ptr<Config> config):
    GNMTBindingBase(model, config)
    {

    }
    void setEmdeddingIndicesInput(void * data)
    {
        setInputCudaBuffer("Query Embedding Indices", data);
    }

    void setAttentionInput(void * data)
    {
        setInputCudaBuffer("Attention shuffled", data);
    }

    void setEncoderLSTMOutputInput(void * data)
    {
        setInputCudaBuffer("Encoder Output", data);
    }

    void setEncoderKeyTransformInput(void * data)
    {
        setInputCudaBuffer("Encoder Key Transform", data);
    }

    void setInputSequenceLengthsInput(void * data)
    {
        setInputCudaBuffer("Input Sequence Lengths", data);
    }

    void setParentLogProbsInput(void * data)
    {
        setInputCudaBuffer("Scorer_parentLogProbs", data);
    }

    void setLengthPenaltyInput(void * data)
    {
        setInputCudaBuffer("Length_Penalty", data);
    }

    void setSoftmaxBoundsInput(void * data)
    {
        setInputCudaBuffer("Scorer_softMaxBounds", data);
    }

    void setHiddInput(int layerId, void * data)
    {
        setInputCudaBuffer((std::string("hidd_l") + std::to_string(layerId)).c_str(), data);
    }

    void setCellInput(int layerId, void * data)
    {
        setInputCudaBuffer((std::string("cell_l") + std::to_string(layerId)).c_str(), data);
    }

    void * getAttentionOutput() const
    {
        return getEngineBoundCudaBuffer("Attention_output");
    }

    void * getLogProbsCombinedOutput() const
    {
        return getEngineBoundCudaBuffer("logProbsCombined");
    }

    void * getBeamIndicesOutput() const
    {
        return getEngineBoundCudaBuffer("Scorer_beamIndices");
    }

    void * getParentLogProbsInput() const
    {
        return getEngineBoundCudaBuffer("Scorer_parentLogProbs");
    }

    void * getEmdeddingIndicesInput() const
    {
        return getEngineBoundCudaBuffer("Query Embedding Indices");
    }

    void * getHiddOutput(int layerId) const
    {
        return getEngineBoundCudaBuffer((std::string("hidd_l") + std::to_string(layerId) + "_out").c_str());
    }

    void * getCellOutput(int layerId) const
    {
        return getEngineBoundCudaBuffer((std::string("cell_l") + std::to_string(layerId) + "_out").c_str());
    }

};


class ShufflerBindingBase: public GNMTBindingBase
{
public:
    ShufflerBindingBase(std::shared_ptr<Model> model, std::shared_ptr<Config> config):
    GNMTBindingBase(model, config)
    {

    }

    void setParentBeamIndicesInput(void * data)
    {
        setInputCudaBuffer("Scorer_parent_beam_idx", data);
    }

    void setAttentionInput(void * data)
    {
        setInputCudaBuffer("Attention_output", data);
    }

    void setHiddInput(int layerId, void * data)
    {
        setInputCudaBuffer((std::string("hidd_l") + std::to_string(layerId) + "_out").c_str(), data);
    }

    void setCellInput(int layerId, void * data)
    {
        setInputCudaBuffer((std::string("cell_l") + std::to_string(layerId) + "_out").c_str(), data);
    }

    void* getAttentionShuffledOutput() const
    {
        return getEngineBoundCudaBuffer("Attention_output_shuffled");
    }

    void* getHiddShuffledOutput(int layerId) const
    {
        return getEngineBoundCudaBuffer((std::string("hidd_l") + std::to_string(layerId) + "_out_shuffled").c_str());
    }

    void* getCellShuffledOutput(int layerId) const
    {
        return getEngineBoundCudaBuffer((std::string("cell_l") + std::to_string(layerId) + "_out_shuffled").c_str());
    }

};
