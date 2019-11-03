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

#include <string>
#include <sstream>
#include "tensorrt/laboratory/core/async_compute.h"
#include "tensorrt/laboratory/inference_manager.h"
#include "tensorrt/laboratory/model.h"
#include "tensorrt/laboratory/bindings.h"
#include "GNMTModel.h"
#include "GNMTBindings.h"
#include "query_sample.h"
#include "loadgen.h"
#include "common.h"
using mlperf::QuerySamplesComplete;
using mlperf::QuerySampleResponse;

struct BatchedTranslationTask
{
    std::vector<mlperf::ResponseId> r_ids;
    std::vector<std::string> batchedStrings;
};


namespace trtlab {
namespace TensorRT {
struct GNMTInferRunner: public AsyncComputeWrapper<void(std::shared_ptr<GNMTBindings>&)>
{
public:
    GNMTInferRunner(std::shared_ptr<GNMTModel> model, std::shared_ptr<InferenceManager>
         resources)
        : mModel{model}, mResources{resources}
    {

    }

    GNMTInferRunner(InferRunner&&) = delete;
    GNMTInferRunner& operator=(InferRunner&&) = delete;

    GNMTInferRunner(const GNMTInferRunner&) = delete;
    GNMTInferRunner& operator=(const GNMTInferRunner&) = delete;

    virtual ~GNMTInferRunner() {}

    //!
    //! \brief Running GNMT inference for loadgen
    //!
    template<typename Post>
    auto Infer(BatchedTranslationTask batch, std::shared_ptr<GNMTBindings> bindings, Post post)
    {
        Enqueue(batch, bindings, post);
    }

    //!
    //! \brief Running GNMT inference for benchmark
    //!
    template<typename Post>
    auto Infer(std::vector<std::string> batch, std::shared_ptr<GNMTBindings> bindings, Post post)
    {
        auto compute = Wrap(post);
        auto future = compute->Future();
        EnqueueBench(batch, bindings, compute);
        return future.share();

    }

protected:
    //!
    //! \brief Subrountine for running GNMT inference for loadgen
    //!
    template<typename Post>
    void Enqueue(BatchedTranslationTask batch, std::shared_ptr<GNMTBindings> bindings, Post post)
    {
        std::vector<std::pair<int, int>> sequenceSampleIdAndLength = mModel->copyInput(batch.batchedStrings, bindings, true);

        mModel->translate(batch.batchedStrings.size(), sequenceSampleIdAndLength, bindings, mResources, batch.r_ids, true);

        bindings->reset();

        (post)(bindings);
    }

    //!
    //! \brief Subrountine for running GNMT inference for benchmark
    //!
    template<typename T>
    void EnqueueBench(std::vector<std::string> batch, std::shared_ptr<GNMTBindings> bindings, std::shared_ptr<AsyncCompute<T>> Post)
    {
        Workers("encoder").enqueue([batch, bindings, this, Post]() mutable {

            std::vector<std::pair<int, int>> sequenceSampleIdAndLength = mModel->copyInput(batch, bindings, true);

            // empty respose ids for benchmark
            std::vector<mlperf::ResponseId> rid;
            rid.resize(0);

            mModel->translate(batch.size(), sequenceSampleIdAndLength, bindings, mResources, rid, true);

            bindings->reset();
            bindings.reset();
            (*Post)(bindings);

         });

    }

    inline ThreadPool& Workers(std::string name)
    {
        return mResources->AcquireThreadPool(name);
    }

public:
    const std::shared_ptr<GNMTModel> GetModelSmartPtr() const { return mModel; }

 private:
    std::shared_ptr<GNMTModel> mModel;
    std::shared_ptr<InferenceManager> mResources;
};

}
}
