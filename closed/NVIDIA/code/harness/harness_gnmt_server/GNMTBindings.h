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

#include "GNMTBindingBase.h"

#include "GNMTModel.h"

class GNMTExtraResources{

public:
    GNMTExtraResources() {};

    void registerConfig(std::shared_ptr<Config> config);

    void allocateResources();

protected:
    std::shared_ptr<HostBufferInt32> mEncoderInputEmbeddingIndicesHostBuffer;
    std::shared_ptr<HostBufferInt32> mEncoderInputSequenceLengthsHostBuffer;
    std::shared_ptr<HostBufferInt32> mGeneratorInputSequenceLengthHostBuffer;

    // Host buffers for the BeamSearch component running on CPU
    // Input for the CPU part
    std::shared_ptr<HostBufferFP32> mBeamSearchInputHostLogProbsCombined;
    std::shared_ptr<HostBufferInt32> mBeamSearchInputHostBeamIndices;
    // Output of the CPU part
    std::shared_ptr<HostBufferFP32>  mBeamSearchOutputHostParentLogProbs;
    std::shared_ptr<HostBufferInt32> mBeamSearchOutputHostNewCandidateTokens;
    std::shared_ptr<HostBufferInt32> mBeamSearchOutputHostParentBeamIndices;

    std::shared_ptr<CudaBufferFP32> mGeneratorInputLengthPenaltyBuffer;
    std::shared_ptr<CudaBufferFP32> mGeneratorInitialParentLogProbsBuffer;
    std::shared_ptr<CudaBufferInt32> mGeneratorInitialCandidateTokensBuffer;
    std::shared_ptr<CudaBufferRaw> mGeneratorInitialAttentionBuffer;

    friend class GNMTModel;
    friend class GNMTBindings;

private:
    int mBeamSize{0};
    int mEncoderMaxSeqLen{0};
    int mDecoderSeqLen{0};
    int mMaxBatchSize{0};
    unsigned int mElementSize{0};
    std::vector<float> mLengthPenalties;

    // Supposed to be the same for all engines and configs
    float mMinimalLogProb;
    int mStartTokan;
    int mHiddenSize;
};

class GNMTBindings{

public:
    GNMTBindings(std::shared_ptr<GNMTModel> gnmt, bool profile=false):
    mConfig(gnmt->mConfig),
    mEncoderBinding((gnmt->mEncoder).GetModelSmartPtr(),gnmt->mConfig),
    mGeneratorBinding((gnmt->mGenerator).GetModelSmartPtr(),gnmt->mConfig),
    mShufflerBinding((gnmt->mShuffler).GetModelSmartPtr(),gnmt->mConfig)
    {
        mGeneratorPtrByBatchSize[mConfig->maxBatchSize] = (gnmt->mGenerator).GetModelSmartPtr();
    }

    GNMTBindings(std::shared_ptr<GNMTModel> gnmt, std::map<int, std::shared_ptr<Model>> generatorPtrByBatchSize, bool profile=false):
    mConfig(gnmt->mConfig),
    mEncoderBinding((gnmt->mEncoder).GetModelSmartPtr(),gnmt->mConfig),
    mGeneratorBinding((gnmt->mGenerator).GetModelSmartPtr(),gnmt->mConfig),
    mShufflerBinding((gnmt->mShuffler).GetModelSmartPtr(),gnmt->mConfig),
    mGeneratorPtrByBatchSize(generatorPtrByBatchSize)
    {

    }

    //!
    //! \brief Synchronize and reset all the bindings and release them
    //!
    void reset();

    //!
    //! \brief Create bindings for each BindingBase from buffers, connect different engines and create execution context
    //!
    void createBindingsAndExecutionContext(std::shared_ptr<InferenceManager> resources, std::shared_ptr<GNMTExtraResources> extraResources, size_t total_count);

protected:

    EncoderBindingBase mEncoderBinding;
    GeneratorBindingBase mGeneratorBinding;
    ShufflerBindingBase mShufflerBinding;

    std::shared_ptr<Config> mConfig;

    std::shared_ptr<GNMTExtraResources> mExtraResources;

    void * mGeneratorParentLogProbsInput{nullptr};
    void * mGeneratorEmdeddingIndicesInput{nullptr};

    // For generator dynamic switching
    std::map<int, std::shared_ptr<Model>> mGeneratorPtrByBatchSize;
    std::map<int, std::shared_ptr<SubExecutionContext>> mGeneratorCtxByBatchSize;

    //!
    //! \brief Create bindings for each BindingBase from buffers and connect different engines
    //!
    void createBindings(std::shared_ptr<Buffers> buffers, std::shared_ptr<GNMTExtraResources> extraResources, size_t total_count);

    friend class GNMTModel;

};
