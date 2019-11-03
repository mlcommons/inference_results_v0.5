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

#ifndef GNMT_ENGINE_H
#define GNMT_ENGINE_H

#include <iomanip>
#include <memory>

#include "NvInfer.h"
#include "params.h"
#include "CudaBuffer.h"
#include "TFWeightsManager.h"
#include "common.h"
#include "utils.h"
#include "../common/half.h" // half_float::half

#include <sys/stat.h>   // Create directory to dump tensors
#include <sstream>
#include <ctime>        // embed date/time in debug directory

using namespace nvinfer1;

class GNMTEngine
{
    template <typename T>
    using uptr = std::unique_ptr<T, samplesCommon::InferDeleter>;

    public:
        //!
        //! \brief Constructor
        //!
        //! \note All classes inherriting from GNMTEngine should ensure all essential member variables are initialized
        //! properly in the constructor.
        //! When engines get serialized, the class instances won't serialize their member variables.
        //! Only the TRT engine itself will be serialized. Hence, proper initialization in the construction is required.
        //!
        //! Exceptions to this are:
        //!
        //! * mEngine
        //! * mContext
        //! * mHostBuffers
        //! * mDeviceBuffers
        //! * mBufIndexMap
        //! These will be initialized by the setup() method
        //!
        GNMTEngine(std::string name, std::shared_ptr<Config> config, bool profile):
            mWeightsManager(),
            mName(name),
            mProfiler(1, name),
            mProfile(profile),
            mConfig(config)
        {}


        //!
        //! \brief Destructor does nothing.
        //!
        virtual ~GNMTEngine() {};

        //!
        //! \brief Run the engine.
        //!
        //! \param batchSize Current batch size, could be smaller than the original one due to batch culling
        //! \param encoderSeqLenSlot use the engine at this slot
        //!
        void run(int batchSize, int encoderSeqLenSlot, cudaStream_t stream);

        //!
        //! \brief Setup the TRT engine by either building (see buildEngine()) or loading it (see: loadEngine)
        //!
        //! \note The GNMTEngine is not meaningful without setup() being called
        //!
        //! \param engineDirName Name of the directory where the enginefile resides OR to where the engine will be written
        //! \param loadEngineFromFile Load the engine from a serialized engine file?
        //! \param storeEngineToFile Store the engine to a file?
        //!
        //! \post This function sets up the following variables:
        //! * mEngine (through loadEngine or buildEngine)
        //! * mContext
        //! * mHostBuffers (through allocateBuffers)
        //! * mDeviceBuffers (through allocateBuffers)
        //! * mBufIndexMap (through allocateBuffers)
        //!
        //! \note Technically, we could move all this into the constructor. However this would increase the complexity of
        //! constructor and will complicate debugging.
        //!
        //! \note Do not initializalize any critical member variables here. These should be initialized in the constructor.
        //!
        void setup(std::string engineDirName = {}, bool loadEngineFromFile = false, bool storeEngineToFile = false);

        //!
        //! \brief Virtual function which will be used to configure the network for each subcomponent of GNMT.
        //! \post mEngine
        //!
        virtual void buildEngine();

        //!
        //! Report GPU timing for this GNMTEngine instance, aggregated over all executions
        //!
        void reportAggregateEngineTime(){
            if (mProfile)
                mProfiler.printLayerTimes();
        }

    protected:
        using UniqueNetworkPtr = uptr<nvinfer1::INetworkDefinition>;
        using UniqueBuilderPtr = uptr<nvinfer1::IBuilder>;
        using UniqueEnginePtr = uptr<nvinfer1::ICudaEngine>;

        //!
        //! \brief Gets element count per sample for the tensor specified.
        //!
        //! \param tensorName Tensor name
        //!
        int64_t getElemCountPerSample(const char * tensorName) const;

        //!
        //! \brief Gets size (in bytes) per sample for the tensor specified.
        //!
        //! \param tensorName Tensor name
        //!
        int64_t getBufferSizePerSample(const char * tensorName) const;

        //!
        //! \brief Sets the buffer for the input tensor specified.
        //!
        //! \param tensorName Tensor name
        //! \param data Pointer to the device memory
        //!
        void setInputTensorBuffer(const char * tensorName, const void * data);

        //!
        //! \brief Gets the buffer for the output tensor specified.
        //!
        //! \param tensorName Tensor name
        //!
        std::shared_ptr<CudaBufferRaw> getOutputTensorBuffer(const char * tensorName) const;

        //!
        //! \brief Pure virtual function which will be used to import weights into the class.
        //!
        //! \note It is recommended to use the TFWeightsManager to load the weights for GNMT
        //! \note The user can add any custom behavior they desire in this function.
        //!
        //! \note Do not initialize/change any member variables from here,
        //!  as they will not be serialized to file when calling storeEngine.
        //!
        virtual void importWeights() = 0;

        //!
        //! \brief Pure virtual function which will be used to configure the network for each subcomponent of GNMT.
        //!
        //! \param network The network that the user wishes to configure.
        //! \param encoderMaxSeqLen Build the network assuming this maximum sequence length.
        //!
        //! \note Do not initialize/change any member variables from here,
        //!  as they will not be serialized to file when calling storeEngine.
        //!
        //! \note A builder and network needs to be created before calling this function.
        //!
        virtual void configureNetwork(nvinfer1::INetworkDefinition* network, int encoderMaxSeqLen) = 0;


        //!
        //! \brief Override this virtual function to do int8-specific configuration.
        //!
        virtual void configureInt8Mode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network);

        //!
        //! \brief Override this virtual function to create a calibration cache that can be used in int8 mode
        //!
        virtual void configureCalibrationCacheMode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network);

        //!
        //! \brief Load the engine from engineFName
        //! \post mEngine
        //!
        void loadEngine(std::string engineFName);

        //!
        //! \brief Store the engine to engineFName
        //!
        void storeEngine(std::string engineFName);

        //!
        //! \brief Based on the engine bindings, this function allocates device buffers for output tensors automatically.
        //! \post mDeviceBuffers
        //!
        void allocateOutputBuffers();

        //!
        //! \brief Copy contents of a tensor for mDeviceBuffers to mHostBuffers
        //! \param tensorName Name of the tensor to be copied
        //! \return Pair of:
        //!     * index number of tensorName in mHostBuffers
        //!     * number of elements in the buffer
        //!
        //! \note This function is not const because it uses the host buffer associated with the CudaBuffer to
        //!       memcopy the values from the device to the host before dumping them to a file.
        //!
        std::pair<int, size_t> copyDeviceTensorToHost(const std::string& tensorName);

        //!
        //! \brief Based on the engine bindings, get binding index for give tensor name.
        //!
        int getBindingIndex(const std::string& tensorName, int encoderSeqLenSlot);

        //!
        //! \brief Mark a specific tensor to be printed if isValidationDumpEnabled() is true
        //! \param network The network the user is configuring
        //! \param tensor The pointer to the tensor that we would like to dump, this tensor will be replaced by this function the caller is advised to use the replaced version to make sure the plugin is not discarded by TensorRT optimizer
        //! \param name What we would like to name the tensor
        //!
        //! \note This adds the tensor name to mDbgTensorNames
        void addDebugTensor(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name);

        TFWeightsManager mWeightsManager;
        std::map<std::string, nvinfer1::Weights> mProcessedWeightsMap;
        std::vector<UniqueEnginePtr> mEngines;
        std::map<std::string, std::shared_ptr<CudaBufferRaw>> mOutBufferMap;
        std::string mName;
        Profiler mProfiler;
        bool mProfile;
        std::vector<IExecutionContext*> mContexts;
        std::shared_ptr<Config> mConfig;
        std::vector<std::vector<void*>> mBindings;
};

#endif
