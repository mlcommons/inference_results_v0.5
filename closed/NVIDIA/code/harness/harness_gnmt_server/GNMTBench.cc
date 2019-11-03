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

#include "GNMTBench.h"

std::unique_ptr<InferBenGNMTBench::Results> InferBenGNMTBench::Run(std::shared_ptr<GNMTModel> model, uint32_t batch_size, std::shared_ptr<GNMTExtraResources> extraResources,
                                    double seconds)
{
    std::vector<std::shared_ptr<GNMTModel>> models = {model};
    return std::move(Run(models, batch_size, extraResources, seconds));
}

std::unique_ptr<InferBenGNMTBench::Results> InferBenGNMTBench::Run(const ModelsList& models, uint32_t batch_size, std::shared_ptr<GNMTExtraResources> extraResources, double seconds)
{
    size_t batch_count = 0;
    std::vector<std::shared_future<void>> futures;
    futures.reserve(1024*1024*1024);

    // Check ModelsList to ensure the requested batch_size is appropriate
    for(const auto& model : models)
    {
        CHECK_LE(batch_size, model->GetMaxBatchSize());
    }

    // Setup std::chrono deadline - no more elapsed lambda
    auto start = std::chrono::high_resolution_clock::now();
    auto last = start + std::chrono::milliseconds(static_cast<long>(seconds * 1000));
    GNMTInferRunner runner(models[0], mResources);

    while(std::chrono::high_resolution_clock::now() < last && ++batch_count)
    {
        size_t model_idx = batch_count % models.size();
        const auto& model = models[model_idx];

        auto gnmtBindings = std::make_shared<GNMTBindings>(runner.GetModelSmartPtr());
        gnmtBindings->createBindingsAndExecutionContext(mResources, extraResources, batch_size);

        std::vector<std::string> batch;
        batch.resize(batch_size);

        for (size_t i = 0; i < batch_size; i++)
        {
            batch.at(i) = sentences[rand() % (sentences.size()-1)];
        }

        futures.push_back(runner.Infer(
            batch,

            gnmtBindings,

            [](std::shared_ptr<GNMTBindings> &bindings) {
                // bindings->reset();
            }

        ));

    }

    // Join worker threads
    for(const auto& f : futures)
    {
        f.wait();
    }

    auto total_time =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    auto inferences = batch_count * batch_size;

    auto results_ptr = std::make_unique<InferBenGNMTBench::Results>();
    Results &results = *results_ptr;
    results[kBatchSize] = batch_size;
    results[kMaxExecConcurrency] = mResources->MaxExecConcurrency();
    results[kMaxCopyConcurrency] = mResources->MaxCopyConcurrency();
    results[kBatchesComputed] = batch_count;
    results[kWalltime] = total_time;
    results[kBatchesPerSecond] = batch_count / total_time;
    results[kInferencesPerSecond] = inferences / total_time;
    results[kExecutionTimePerBatch] =
        total_time / (batch_count);

    DLOG(INFO) << "Benchmark Run Complete";

    DLOG(INFO) << "Inference Results: " << results[kBatchesComputed] << " batches computed in "
              << results[kWalltime] << " seconds on " << results[kMaxExecConcurrency]
              << " compute streams using batch_size: " << results[kBatchSize]
              << "; inf/sec: " << results[kInferencesPerSecond]
              << "; batches/sec: " << results[kBatchesPerSecond]
              << "; execution time per batch: " << results[kExecutionTimePerBatch];

    return std::move(results_ptr);
}
