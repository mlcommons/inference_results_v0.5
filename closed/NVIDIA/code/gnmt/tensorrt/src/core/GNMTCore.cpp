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

#include "GNMTCore.h"

#include "GNMTBeamSearch.h"

#include <algorithm>

void GNMTCore::reportAggregateEngineTime(){
    if (mProfile){
        mEncoder.reportAggregateEngineTime();
        mGenerator.reportAggregateEngineTime();
        mShuffler.reportAggregateEngineTime();
        printf("\nScorer takes %.3fs on CPU\n", mScorerCpuTimeTotal);
    }
}

void GNMTCore::buildCalibrationCache(){
    if(! mConfig->buildCalibrationCache()){
        std::cerr << "Graph not configured to build calibration cache" << std::endl;
    }

    mGenerator.buildEngine();
}

void GNMTCore::setup(){
    mEncoder.setup(mEngineDir, mLoadEngine, mStoreEngine);

    mEncoderInputEmbeddingIndicesBuffer = std::make_shared<CudaBufferInt32>(mConfig->encoderMaxSeqLengths[0] * mConfig->maxBatchSize);
    mEncoderInputSequenceLengthsBuffer = std::make_shared<CudaBufferInt32>(mConfig->maxBatchSize);

    mEncoderInputEmbeddingIndicesHostBuffer = std::make_shared<HostBufferInt32>(mConfig->encoderMaxSeqLengths[0] * mConfig->maxBatchSize);
    mEncoderInputSequenceLengthsHostBuffer = std::make_shared<HostBufferInt32>(mConfig->maxBatchSize);

    mShuffler.setup(mEngineDir, mLoadEngine, mStoreEngine);

    mShufflerInputParentBeamIndicesBuffer = std::make_shared<CudaBufferInt32>(mConfig->maxBatchSize * mConfig->beamSize);

    mGenerator.setup(mEngineDir, mLoadEngine, mStoreEngine);

    mGeneratorInitialCandidateTokensBuffer = std::make_shared<CudaBufferInt32>(std::vector<int>(mConfig->maxBatchSize * mConfig->beamSize, mConfig->START_TOKEN));
    mGeneratorInitialAttentionBuffer = std::make_shared<CudaBufferRaw>(mConfig->beamSize * mConfig->hiddenSize * mConfig->maxBatchSize * samplesCommon::getElementSize(mConfig->prec));
    mGeneratorInitialAttentionBuffer->fillWithZero();

    std::vector<float> initialParentLogProbs(mConfig->maxBatchSize * mConfig->beamSize, mConfig->minimalLogProb);
    // Initialize the first element only of each beam to 0, others are set to very low log prob value
    for(int i = 0; i < mConfig->maxBatchSize; ++i)
        initialParentLogProbs[i * mConfig->beamSize] = 0.0F;
    mGeneratorInitialParentLogProbsBuffer = std::make_shared<CudaBufferFP32>(initialParentLogProbs);

    mGeneratorInputSequenceLengthHostBuffer = std::make_shared<HostBufferInt32>(mConfig->beamSize * mConfig->maxBatchSize);

    mGeneratorInputIndicesBuffer = std::make_shared<CudaBufferInt32>(mConfig->beamSize * mConfig->maxBatchSize);
    mGeneratorInputSequenceLengthBuffer = std::make_shared<CudaBufferInt32>(mConfig->beamSize * mConfig->maxBatchSize);
    mGeneratorInputParentCombinedLikelihoodsBuffer = std::make_shared<CudaBufferFP32>(mConfig->beamSize * mConfig->maxBatchSize);
    std::vector<float> lengthPenalties((mConfig->decoderSeqLen + 1) * mConfig->maxBatchSize);
    for(int i = 0; i < (mConfig->decoderSeqLen + 1); ++i)
        lengthPenalties[i] = mConfig->getLengthPenalyMultiplier(i);
    mGeneratorInputLengthPenaltyBuffer = std::make_shared<CudaBufferFP32>(lengthPenalties);

    mBeamSearchInputHostLogProbsCombined = std::make_shared<HostBufferFP32>(mConfig->maxBatchSize * mConfig->beamSize);
    mBeamSearchInputHostBeamIndices = std::make_shared<HostBufferInt32>(mConfig->maxBatchSize * mConfig->beamSize);
    mBeamSearchOutputHostParentLogProbs = std::make_shared<HostBufferFP32>(mConfig->maxBatchSize * mConfig->beamSize);
    mBeamSearchOutputHostNewCandidateTokens = std::make_shared<HostBufferInt32>(mConfig->maxBatchSize * mConfig->beamSize);
    mBeamSearchOutputHostParentBeamIndices = std::make_shared<HostBufferInt32>(mConfig->maxBatchSize * mConfig->beamSize);


    // Initialize input buffers for the engines
    mEncoder.setEmdeddingIndicesInput(mEncoderInputEmbeddingIndicesBuffer->data());
    mEncoder.setInputSequenceLengthsInput(mEncoderInputSequenceLengthsBuffer->data());

    mShuffler.setParentBeamIndicesInput(mShufflerInputParentBeamIndicesBuffer->data());
    mShuffler.setAttentionInput(mGenerator.getAttentionOutput()->data());
    for(int i = 0; i < mConfig->decoderLayerCount; ++i)
    {
        mShuffler.setHiddInput(i, mGenerator.getHiddOutput(i)->data());
        mShuffler.setCellInput(i, mGenerator.getCellOutput(i)->data());
    }

    mGenerator.setEncoderLSTMOutputInput(mEncoder.getLSTMOutput()->data());
    mGenerator.setEncoderKeyTransformInput(mEncoder.getKeyTransformOutput()->data());
    mGenerator.setInputSequenceLengthsInput(mGeneratorInputSequenceLengthBuffer->data());

    // If we are serializing the engines, dump out hyperparameters
    if (mStoreEngine){
        mConfig->writeToJSON(mEngineDir + "/" + mConfigFileName);
    }

    mScorerCpuTimeTotal = 0.0;
}

std::vector<std::vector<std::string>> GNMTCore::translate(std::vector<std::string> batch, bool batchCulling){
    int actualBatchSize = batch.size();

    // Ensure that we are not exceeding the max batch size
    assert(actualBatchSize <= mConfig->maxBatchSize);

    // TBD: To avoid allocating space over and over, we should create an mTokenIndices
    // vector of size mConfig->maxBatchSize vectors with size mConfig->maxEncoderSeqLength each.
    // This will change the mechanics a bit as we would need to have mIndexer fill the encoderSeqLenVector
    // simultaneously as the mTokenIndices vector. (right now we use tokenIndices[i].size() to fill in encoderSeqLenVector)
    std::vector<vector<unsigned int>> tokenIndices(actualBatchSize, std::vector<unsigned int>());

    // Find the indices of each sentence in the batch
    mIndexer.findIndices(batch, tokenIndices);

    std::vector<std::vector<std::string>> tokenWords;
    translate(tokenIndices, tokenWords, batchCulling);

    return tokenWords;
}

void GNMTCore::translate(const std::vector<vector<unsigned int>>& tokenIndices, std::vector<vector<std::string>>& tokenWords, bool batchCulling){

    int actualBatchSize = tokenIndices.size();
    std::vector<std::pair<int, int>> sequenceSampleIdAndLength = GNMTCoreUtil::sortBatch(tokenIndices, batchCulling);

    int exactEncoderMaxSeqLen = std::max_element(sequenceSampleIdAndLength.begin(), sequenceSampleIdAndLength.end(), [](const std::pair<int, int>& x, const std::pair<int, int>& y) { return x.second < y.second; })->second;
    int bestEncoderSeqLenSlot = mConfig->getBestEncoderSeqLenSlot(exactEncoderMaxSeqLen);
    int encoderMaxSeqLen = mConfig->encoderMaxSeqLengths[bestEncoderSeqLenSlot];

    // Calculate sequence lengths and copy over input tokens
    // Note that sequence length calculation is very sensitive to final output
    // Sequence lengths should not include the STOP_TOKEN, as this would
    // artificially add one timestep to the LSTMs
    GNMTCoreUtil::calculateSeqLengths(tokenIndices, sequenceSampleIdAndLength, mEncoderInputSequenceLengthsHostBuffer, mEncoderInputEmbeddingIndicesHostBuffer, encoderMaxSeqLen, mConfig->STOP_TOKEN);

    // use default stream
    cudaStream_t stream = 0;
    
    // Copying input for the Encoder to GPU
    CHECK_CUDA(cudaMemcpyAsync(
        mEncoderInputEmbeddingIndicesBuffer->data(),
        mEncoderInputEmbeddingIndicesHostBuffer->data(),
        encoderMaxSeqLen * actualBatchSize * sizeof(int),
        cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(
        mEncoderInputSequenceLengthsBuffer->data(),
        mEncoderInputSequenceLengthsHostBuffer->data(),
        actualBatchSize * sizeof(int),
        cudaMemcpyHostToDevice, stream));

    // Run the encoder
    mEncoder.run(actualBatchSize, bestEncoderSeqLenSlot, stream);

    // Replicating input sequence lengths beam-size times and copy them to GPU
    for (int i = 0; i < actualBatchSize; ++i)
    {
        std::fill(
            mGeneratorInputSequenceLengthHostBuffer->data() + (mConfig->beamSize * i),
            mGeneratorInputSequenceLengthHostBuffer->data() + (mConfig->beamSize * (i + 1)),
            sequenceSampleIdAndLength[i].second);
    }
    CHECK_CUDA(cudaMemcpyAsync(
        mGeneratorInputSequenceLengthBuffer->data(),
        mGeneratorInputSequenceLengthHostBuffer->data(),
        mConfig->beamSize * actualBatchSize * sizeof(int),
        cudaMemcpyHostToDevice, stream));

    const size_t maxDecoderSeqLen = *std::max_element(mEncoderInputSequenceLengthsHostBuffer->data(), mEncoderInputSequenceLengthsHostBuffer->data() + actualBatchSize) * 2;

    BeamSearch beamSearch(actualBatchSize, mConfig);

    int generatorBatchSize = actualBatchSize;
    // Run Generator for each token
    for (int tok = 0; tok < mConfig->decoderSeqLen; ++tok)
    {
        if (tok > 0)
        {
            // Run the shuffler starting from the second iteration.
            mShuffler.run(generatorBatchSize, bestEncoderSeqLenSlot, stream);
        }

        if (tok == 0)
        {
            // Initialize input buffers for the generator for the 1st iteration
            mGenerator.setEmdeddingIndicesInput(mGeneratorInitialCandidateTokensBuffer->data());
            mGenerator.setAttentionInput(mGeneratorInitialAttentionBuffer->data());
            mGenerator.setParentLogProbsInput(mGeneratorInitialParentLogProbsBuffer->data());
            for(int i = 0; i < mConfig->decoderLayerCount; ++i)
            {
                mGenerator.setHiddInput(i, mEncoder.getHiddOutput(i)->data());
                mGenerator.setCellInput(i, mEncoder.getCellOutput(i)->data());
            }
        }
        else if (tok == 1)
        {
            // Initialize input buffers for the generator to get data from the shuffler
            mGenerator.setEmdeddingIndicesInput(mGeneratorInputIndicesBuffer->data());
            mGenerator.setAttentionInput(mShuffler.getAttentionShuffledOutput()->data());
            mGenerator.setParentLogProbsInput(mGeneratorInputParentCombinedLikelihoodsBuffer->data());
            for(int i = 0; i < mConfig->decoderLayerCount; ++i)
            {
                mGenerator.setHiddInput(i, mShuffler.getHiddShuffledOutput(i)->data());
                mGenerator.setCellInput(i, mShuffler.getCellShuffledOutput(i)->data());
            }
        }
        mGenerator.setLengthPenaltyInput(mGeneratorInputLengthPenaltyBuffer->data() + tok);

        // Run the generator
        mGenerator.run(generatorBatchSize, bestEncoderSeqLenSlot, stream);

        CHECK_CUDA(cudaMemcpyAsync(
            mBeamSearchInputHostLogProbsCombined->data(),
            mGenerator.getLogProbsCombinedOutput()->data(),
            generatorBatchSize * mConfig->beamSize * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(
            mBeamSearchInputHostBeamIndices->data(),
            mGenerator.getBeamIndicesOutput()->data(),
            generatorBatchSize * mConfig->beamSize * sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        double t0 = seconds();
        int unfinishedSampleCount = beamSearch.run(
            generatorBatchSize,
            maxDecoderSeqLen,
            mBeamSearchInputHostLogProbsCombined->data(),
            mBeamSearchInputHostBeamIndices->data(),
            mBeamSearchOutputHostParentLogProbs->data(),
            mBeamSearchOutputHostNewCandidateTokens->data(),
            mBeamSearchOutputHostParentBeamIndices->data());
        double cpuTime = seconds() - t0;
        mScorerCpuTimeTotal += cpuTime;

        // Check if the beam search finished for all the samples in the batch
        if (unfinishedSampleCount == 0)
            break;

        CHECK_CUDA(cudaMemcpyAsync(
            mGeneratorInputParentCombinedLikelihoodsBuffer->data(),
            mBeamSearchOutputHostParentLogProbs->data(),
            generatorBatchSize * mConfig->beamSize * sizeof(float),
            cudaMemcpyHostToDevice,stream));
        CHECK_CUDA(cudaMemcpyAsync(
            mGeneratorInputIndicesBuffer->data(),
            mBeamSearchOutputHostNewCandidateTokens->data(),
            generatorBatchSize * mConfig->beamSize * sizeof(int32_t),
            cudaMemcpyHostToDevice,stream));
        CHECK_CUDA(cudaMemcpyAsync(
            mShufflerInputParentBeamIndicesBuffer->data(),
            mBeamSearchOutputHostParentBeamIndices->data(),
            generatorBatchSize * mConfig->beamSize * sizeof(int32_t),
            cudaMemcpyHostToDevice,stream));

        if (batchCulling)
            generatorBatchSize = unfinishedSampleCount;
    }
    // End of Generator loop

    // Get the final prediction for the batch.
    vector<vector<int>> finalPredictions(actualBatchSize);
    for (int i = 0; i < actualBatchSize; i++)
    {
        int sampleId = sequenceSampleIdAndLength[i].first;
        finalPredictions[sampleId] = beamSearch.getFinalPrediction(i);
    }

    // Get the German words for the predicted tokens
    mIndexer.findWords(tokenWords, finalPredictions, RuntimeInfo::currentBatch);
}
