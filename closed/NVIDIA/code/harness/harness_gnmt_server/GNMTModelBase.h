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

#include "tensorrt/laboratory/core/affinity.h"
#include "tensorrt/laboratory/core/thread_pool.h"
#include "tensorrt/laboratory/infer_bench.h"
#include "tensorrt/laboratory/infer_runner.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/bindings.h"
#include "tensorrt/laboratory/buffers.h"
#include "tensorrt/laboratory/core/memory/memory_stack.h"
#include "tensorrt/laboratory/core/memory/descriptor.h"
#include "tensorrt/laboratory/cuda/memory/cuda_device.h"
#include "tensorrt/laboratory/inference_manager.h"


using trtlab::TensorRT::Model;
using trtlab::TensorRT::Bindings;
using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::Buffers;
using trtlab::Memory::MemoryStack;
using trtlab::Memory::Descriptor;
using trtlab::Memory::CudaDeviceMemory;
using trtlab::TensorRT::InferenceManager;
using trtlab::TensorRT::ExecutionContext;

using namespace nvinfer1;

class GNMTModelBase
{
public:
    GNMTModelBase(std::string name, std::shared_ptr<Config> config, bool profile):
        mName(name),
        mProfiler(1, name),
        mProfile(profile),
        mConfig(config)
    {}

    virtual ~GNMTModelBase() {};

    //!
    //! \brief Load TRT-Lab model
    //!
    void loadTRTLabModel(std::shared_ptr<Model> model)
    {
        mModel = model;
    }

    //!
    //! \brief Get shared model pointer
    //!
    std::shared_ptr<Model> GetModelSmartPtr() const { return mModel; }

    //!
    //! \brief Set the device id for this engine
    //!
    void setDevice(int device)
    {
        mDevice = device;
    }

    //!
    //! \brief Get the device id associated with this engine
    //!
    int getDevice()
    {
        return mDevice;
    }

    //!
    //! \brief Get path to engine file
    //!
    const std::string getEngineFName(std::string engineDirName, int currentSeqLen)
    {
        return engineDirName + "/" + mName + ".engine" + ".encseqlen_" + std::to_string(currentSeqLen);
    }

    //!
    //! \brief Deserialize engines and setup GNMTModelBase
    //!
    //! \param[in] engineDirName Engine directory name specifying where to look for engine files
    //! \param[in] runtime TRT-Lab Runtime (Wrapper of TensorRT runtime) to deserialize engine
    //! \param[in] max_batch_size_per_gnmt Max batch size of this gnmt model
    //! \param[in] engine_names Vector of all sub engine names
    //! \param[in] models Vector of all sub models
    //! \param[in] device Device id specifying the gpu it is running on
    //! 
    //!
    void setup(std::string engineDirName, std::shared_ptr<Runtime> runtime, int &max_batch_size_per_gnmt, std::vector<string> &engine_names, std::vector<std::shared_ptr<Model>> &models, int device, int currentSeqLen)
    {
        // Setup engine file name if requested
        const std::string file = getEngineFName(engineDirName, currentSeqLen);

        setDevice(device);
        CHECK_EQ(cudaSetDevice(mDevice), CUDA_SUCCESS) << "fail to launch device " << 1;

        auto model = runtime->DeserializeEngine(file);

        loadTRTLabModel(model);

        model->SetName(file);

        engine_names.push_back(file);
        models.push_back(model);

        max_batch_size_per_gnmt =
              std::max(max_batch_size_per_gnmt, model->GetMaxBatchSize());
    }

    //!
    //! \brief Wrapper function for running TRT engines
    //!
    virtual void run(int batchSize, GNMTBindingBase bindings, std::shared_ptr<InferenceManager> resources)
    {
        _run(batchSize, bindings.getBindings(), bindings.getExecutionContext());
    }

    //!
    //! \brief Report GPU timing for this GNMTEngine instance, aggregated over all executions
    //!
    void reportAggregateEngineTime(){
        if (mProfile)
            mProfiler.printLayerTimes();
    }

protected:
    std::shared_ptr<Model> mModel;
    int mDevice{0};
    std::string mName;
    Profiler mProfiler;
    bool mProfile;
    std::shared_ptr<Config> mConfig;

    //!
    //! \brief Running TRT engines
    //! 
    //! \param bindings Input and output buffers
    //! \param ctx Execution contexts
    //!
    //! \note Before running, it grabs execution contexts of this engine from resources.
    //! \note But execution contexts comes from a pool and we need to Synchronize before release it back to the pool
    //! \note Otherwise, the behavior may be undefined because the device memory of a execution context also comes from a pool
    //!
    void _run(int batchSize, std::shared_ptr<Bindings> bindings, std::shared_ptr<SubExecutionContext> ctx)
    {
        bindings->SetBatchSize(batchSize);

        CHECK_EQ(cudaSetDevice(mDevice), CUDA_SUCCESS) << "fail to launch device " << 1;
        
        ctx->Infer(bindings);
    }
};
