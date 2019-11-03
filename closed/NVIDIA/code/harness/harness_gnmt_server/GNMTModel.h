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
#include "GNMTCore.h"
#include "GNMTBeamSearch.h"
#include "GNMTModelBase.h"
#include "utils.h"
#include "params.h"
#include <random>

#include "tensorrt/laboratory/core/affinity.h"
#include "tensorrt/laboratory/core/thread_pool.h"
#include "tensorrt/laboratory/infer_bench.h"
#include "tensorrt/laboratory/infer_runner.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/bindings.h"
#include "tensorrt/laboratory/buffers.h"

using trtlab::TensorRT::Runtime;
using trtlab::TensorRT::Bindings;
using trtlab::TensorRT::Buffers;

#include "query_sample.h"
#include "loadgen.h"
using mlperf::QuerySamplesComplete;
using mlperf::QuerySampleResponse;

class GNMTBindings;
class GNMTModel{
public:

    //!
    //! \brief Initializes all GNMTModelBase classes that compose GNMT
    //! \note This class does not hold any memory needed for TRT execution
    //!
    GNMTModel(std::shared_ptr<Config> config, std::string engineDir="", bool profile=false):
        mConfig(config),
        mEngineDir(engineDir),
        mIndexer(config),
        mEncoder("Encoder", config, profile),
        mShuffler("Shuffler", config, profile),
        mGenerator("Generator", config, profile),
        mProfile(profile) {;}

    //!
    //! \brief Deserialize all the engines in GNMT and do proper setup
    //!
    void setup(std::shared_ptr<Runtime> runtime, int device, int &max_batch_size_per_gnmt, std::vector<string> &engine_names, std::vector<std::shared_ptr<Model>> &models, int currentSeqLen){

        mEncoder.setup(mEngineDir, runtime, max_batch_size_per_gnmt, engine_names, models, device, currentSeqLen);

        mGenerator.setup(mEngineDir, runtime, max_batch_size_per_gnmt, engine_names, models, device, currentSeqLen);

        mShuffler.setup(mEngineDir, runtime, max_batch_size_per_gnmt, engine_names, models, device, currentSeqLen);

        // Reset mScorerCpuTimeTotal to increase timing accuracy
        mScorerCpuTimeTotal = 0.0;
    }

    //!
    //! \brief Translate input text and return output text
    //! \param batch A batch of samples in a source language (e.g., English)
    //! \param bindings Input and output bindings for all the engines in GNMT
    //! \param resources It holds execution contexts for engine exeuction 
    //! \param batchCulling Improve translation speed by reducing the batch size when sentences finish
    //!
    std::vector<std::pair<int, int>> copyInput(std::vector<std::string> batch, std::shared_ptr<GNMTBindings> bindings, bool batchCulling);
    void translate(int actualBatchSize, std::vector<std::pair<int, int>> sequenceSampleIdAndLength,  std::shared_ptr<GNMTBindings> bindings, std::shared_ptr<InferenceManager> resources, std::vector<mlperf::ResponseId> rids, bool batchCulling);
    
    //!
    //! \brief Report the aggregated runtimes on a per-engine basis
    //!
    void reportAggregateEngineTime();

    //!
    //! \brief Get the max batch size from config
    //!
    int GetMaxBatchSize()
    {
        return mConfig->maxBatchSize;
    }

    //!
    //! \brief Get shared model pointer
    //!
    std::shared_ptr<Model> GetGeneratorSmartPtr() const { return mGenerator.GetModelSmartPtr(); }
protected:
    //!
    //! Engines wrapped under GNMTModelBase in GNMT
    //!
    Indexer mIndexer;
    GNMTModelBase mEncoder;
    GNMTModelBase mShuffler;
    GNMTModelBase mGenerator;

    //!
    //! Pointer to the configuration of GNMT
    //!
    std::shared_ptr<Config> mConfig;

private:
    //!
    //! Serialization related data
    //!
    std::string mEngineDir;

    //! Runtime of the CPU part of the scorer
    double mScorerCpuTimeTotal = 0.0;
    bool mProfile;

    //! Filename of configuration file in case we will serialize it
    const std::string mConfigFileName {"config.json"};

    friend class GNMTBindings;
};
