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

#include "GNMTBindings.h"

void GNMTExtraResources::registerConfig(std::shared_ptr<Config> config)
{
    mBeamSize = std::max(mBeamSize, config->beamSize);
    mEncoderMaxSeqLen = std::max(mEncoderMaxSeqLen, config->encoderMaxSeqLengths[0]);
    mDecoderSeqLen = std::max(mDecoderSeqLen, config->decoderSeqLen);
    mMaxBatchSize = std::max(mMaxBatchSize, config->maxBatchSize);
    mElementSize = std::max(mElementSize, samplesCommon::getElementSize(config->prec));

    mLengthPenalties.resize((mDecoderSeqLen + 1) * mMaxBatchSize);
    for(int i = 0; i < (mDecoderSeqLen + 1); ++i)
        mLengthPenalties[i] = config->getLengthPenalyMultiplier(i);

    // Same for all configs
    mMinimalLogProb = config->minimalLogProb;
    mStartTokan = config->START_TOKEN;
    mHiddenSize = config->hiddenSize;
}

void GNMTExtraResources::allocateResources()
{
    // Create host buffers
    mEncoderInputEmbeddingIndicesHostBuffer = std::make_shared<HostBufferInt32>(mEncoderMaxSeqLen *  mMaxBatchSize);
    mEncoderInputSequenceLengthsHostBuffer = std::make_shared<HostBufferInt32>(mMaxBatchSize);

    mGeneratorInputSequenceLengthHostBuffer = std::make_shared<HostBufferInt32>(mBeamSize * mMaxBatchSize);

    mBeamSearchInputHostLogProbsCombined = std::make_shared<HostBufferFP32>(mMaxBatchSize * mBeamSize);
    mBeamSearchInputHostBeamIndices = std::make_shared<HostBufferInt32>(mMaxBatchSize * mBeamSize);
    mBeamSearchOutputHostParentLogProbs = std::make_shared<HostBufferFP32>(mMaxBatchSize * mBeamSize);
    mBeamSearchOutputHostNewCandidateTokens = std::make_shared<HostBufferInt32>(mMaxBatchSize * mBeamSize);
    mBeamSearchOutputHostParentBeamIndices = std::make_shared<HostBufferInt32>(mMaxBatchSize * mBeamSize);

    mGeneratorInputLengthPenaltyBuffer = std::make_shared<CudaBufferFP32>(mLengthPenalties);

    mGeneratorInitialCandidateTokensBuffer = std::make_shared<CudaBufferInt32>(std::vector<int>(mMaxBatchSize * mBeamSize, mStartTokan));
    
    mGeneratorInitialAttentionBuffer = std::make_shared<CudaBufferRaw>(mBeamSize * mHiddenSize * mMaxBatchSize * mElementSize);
    mGeneratorInitialAttentionBuffer->fillWithZero();

    std::vector<float> initialParentLogProbs(mMaxBatchSize * mBeamSize, mMinimalLogProb);
    // Initialize the first element only of each beam to 0, others are set to very low log prob value
    for(int i = 0; i < mMaxBatchSize; ++i)
        initialParentLogProbs[i * mBeamSize] = 0.0F;
    mGeneratorInitialParentLogProbsBuffer = std::make_shared<CudaBufferFP32>(initialParentLogProbs);
}

void GNMTBindings::createBindings(std::shared_ptr<Buffers> buffers, std::shared_ptr<GNMTExtraResources> extraResources, size_t total_count)
{
    // Encoder
    mEncoderBinding.createBindings(buffers, total_count);

    // Generator
    mGeneratorBinding.createBindings(buffers, total_count);

    // Generator: setAttentionInput
    mGeneratorBinding.setAttentionInput(extraResources->mGeneratorInitialAttentionBuffer->data());

    // Generator: setParentLogProbsInput
    mGeneratorParentLogProbsInput = mGeneratorBinding.getParentLogProbsInput();
    mGeneratorBinding.setParentLogProbsInput(extraResources->mGeneratorInitialParentLogProbsBuffer->data());

    // Generator: setEmdeddingIndicesInput
    mGeneratorEmdeddingIndicesInput = mGeneratorBinding.getEmdeddingIndicesInput();
    mGeneratorBinding.setEmdeddingIndicesInput(extraResources->mGeneratorInitialCandidateTokensBuffer->data());

    // Shuffler
    mShufflerBinding.createBindings(buffers, total_count);

    // Setting up input and output relations
    mShufflerBinding.setAttentionInput(mGeneratorBinding.getAttentionOutput());

    for(int i = 0; i < mConfig->decoderLayerCount; ++i)
    {
        mShufflerBinding.setHiddInput(i, mGeneratorBinding.getHiddOutput(i));
        mShufflerBinding.setCellInput(i, mGeneratorBinding.getCellOutput(i));
    }

    mGeneratorBinding.setEncoderLSTMOutputInput(mEncoderBinding.getLSTMOutput());
    mGeneratorBinding.setEncoderKeyTransformInput(mEncoderBinding.getKeyTransformOutput());

}

void GNMTBindings::createBindingsAndExecutionContext(std::shared_ptr<InferenceManager> resources, std::shared_ptr<GNMTExtraResources> extraResources, size_t total_count)
{
    mExtraResources = extraResources;
    auto buffers = resources->GetBuffers();
    createBindings(buffers, extraResources, total_count);

    auto ctx = resources->GetExecutionContext();

    auto encoderCtx = resources->GetSubExecutionContext(ctx, mEncoderBinding.GetModelSmartPtr());
    mEncoderBinding.setExecutionContext(encoderCtx);

    for (auto const& it : mGeneratorPtrByBatchSize)
    {
        auto generatorCtx = resources->GetSubExecutionContext(ctx, it.second);
        mGeneratorCtxByBatchSize[it.first] = generatorCtx;
    }

    auto shufflerCtx = resources->GetSubExecutionContext(ctx, mShufflerBinding.GetModelSmartPtr());
    mShufflerBinding.setExecutionContext(shufflerCtx);
}

void GNMTBindings::reset(){

    mShufflerBinding.bindingsSynchronize();

    mEncoderBinding.reset();

    mGeneratorBinding.reset();

    mShufflerBinding.reset();

    mExtraResources.reset();
}
