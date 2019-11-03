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

#include "tensorrt/laboratory/infer_bench.h"
#include "GNMTModel.h"
#include "GNMTBindings.h"
#include "GNMTInferRunner.h"
#include <glog/logging.h>

#include <iostream>
#include <vector>
#include <random>
#include <functional> //for std::function
#include <algorithm>  //for std::generate_n

using trtlab::TensorRT::InferBench;
using trtlab::TensorRT::InferBenchKey;
using trtlab::TensorRT::GNMTInferRunner;

using trtlab::TensorRT::kMaxExecConcurrency;
using trtlab::TensorRT::kMaxCopyConcurrency;
using trtlab::TensorRT::kBatchSize;
using trtlab::TensorRT::kWalltime;
using trtlab::TensorRT::kBatchesComputed;
using trtlab::TensorRT::kBatchesPerSecond;
using trtlab::TensorRT::kInferencesPerSecond;
using trtlab::TensorRT::kSecondsPerBatch;
using trtlab::TensorRT::kExecutionTimePerBatch;

size_t get_string_length(std::string input);

class InferBenGNMTBench: public InferBench
{
public:
    //!
    //! \brief InferBenGNMTBench constructor
    //!
    InferBenGNMTBench(std::shared_ptr<InferenceManager> resources, std::vector<std::string> sentences): mResources(resources), InferBench(resources), sentences(sentences) {};
    ~InferBenGNMTBench() {};

    using ModelsList = std::vector<std::shared_ptr<GNMTModel>>;
    using Results = std::map<InferBenchKey, double>;

    //!
    //! \brief Running GNMT benchmarks
    //! \param[in] model GNMTmodel pointer
    //! \param[in] batch_size The batch size to run benchmark with
    //! \param[in] seconds Time limit to run GNMT benchmarks
    //!
    //! \return Benchmark results pointer
    std::unique_ptr<Results> Run(const std::shared_ptr<GNMTModel> model, uint32_t batch_size, std::shared_ptr<GNMTExtraResources> extraResources, double seconds = 5.0);

    //!
    //! \brief Running GNMT benchmarks with multiple models
    //!
    std::unique_ptr<Results> Run(const ModelsList& models, uint32_t batch_size, std::shared_ptr<GNMTExtraResources> extraResources, double seconds = 5.0);
 

private:
  std::shared_ptr<InferenceManager> mResources;
  std::vector<std::string> sentences;

};
