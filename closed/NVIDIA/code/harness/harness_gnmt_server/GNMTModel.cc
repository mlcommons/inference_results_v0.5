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

#include "GNMTModel.h"
#include "GNMTBindings.h"

void GNMTModel::reportAggregateEngineTime(){
    if (mProfile){
        mEncoder.reportAggregateEngineTime();
        mGenerator.reportAggregateEngineTime();
        mShuffler.reportAggregateEngineTime();
        printf("\nScorer takes %.3fs on CPU\n", mScorerCpuTimeTotal);
    }
}

std::vector<std::pair<int, int>> GNMTModel::copyInput(std::vector<std::string> batch, std::shared_ptr<GNMTBindings> bindings, bool batchCulling)
{
    auto EncoderBinding = bindings->mEncoderBinding;
    auto GeneratorBinding = bindings->mGeneratorBinding;
    auto GeneratorInputSequenceLengthHostBuffer = bindings->mExtraResources->mGeneratorInputSequenceLengthHostBuffer;
    auto EncoderInputSequenceLengthsHostBuffer = bindings->mExtraResources->mEncoderInputSequenceLengthsHostBuffer;
    auto EncoderInputEmbeddingIndicesHostBuffer = bindings->mExtraResources->mEncoderInputEmbeddingIndicesHostBuffer;

    int actualBatchSize = batch.size();

    // Ensure that we are not exceeding the max batch size
    CHECK(actualBatchSize <= mConfig->maxBatchSize);

    // TBD: To avoid allocating space over and over, we should create an mTokenIndices
    // vector of size mConfig->maxBatchSize vectors with size mConfig->maxEncoderSeqLength each.
    // This will change the mechanics a bit as we would need to have mIndexer fill the encoderSeqLenVector
    // simultaneously as the mTokenIndices vector. (right now we use tokenIndices[i].size() to fill in encoderSeqLenVector)
    std::vector<vector<unsigned int>> tokenIndices(actualBatchSize, std::vector<unsigned int>());

    mIndexer.findIndices(batch, tokenIndices);

    std::vector<std::pair<int, int>> sequenceSampleIdAndLength = GNMTCoreUtil::sortBatch(tokenIndices, batchCulling);

    // Calculate sequence lengths and copy over input tokens
    // Note that sequence length calculation is very sensitive to final output
    // Sequence lengths should not include the STOP_TOKEN, as this would
    // artificially add one timestep to the LSTMs
    int exactEncoderMaxSeqLen = std::max_element(sequenceSampleIdAndLength.begin(), sequenceSampleIdAndLength.end(), [](const std::pair<int, int>& x, const std::pair<int, int>& y) { return x.second < y.second; })->second;
    int bestEncoderSeqLenSlot = mConfig->getBestEncoderSeqLenSlot(exactEncoderMaxSeqLen);
    int encoderMaxSeqLen = mConfig->encoderMaxSeqLengths[bestEncoderSeqLenSlot];

    GNMTCoreUtil::calculateSeqLengths(tokenIndices, sequenceSampleIdAndLength, EncoderInputSequenceLengthsHostBuffer, EncoderInputEmbeddingIndicesHostBuffer, encoderMaxSeqLen, mConfig->STOP_TOKEN);

    // Copying input for the Encoder to GPU

    EncoderBinding.setInputCudaBufferFromHost("Encoder_embedding_indices", EncoderInputEmbeddingIndicesHostBuffer->data(), encoderMaxSeqLen * actualBatchSize * sizeof(int));
    EncoderBinding.setInputCudaBufferFromHost("Encoder Sequence Lengths", EncoderInputSequenceLengthsHostBuffer->data(), actualBatchSize * sizeof(int));

    for (int i = 0; i < actualBatchSize; ++i)
    {
        std::fill(
            GeneratorInputSequenceLengthHostBuffer->data() + (mConfig->beamSize * i),
            GeneratorInputSequenceLengthHostBuffer->data() + (mConfig->beamSize * (i + 1)),
            sequenceSampleIdAndLength[i].second);
    }

    GeneratorBinding.setInputCudaBufferFromHost("Input Sequence Lengths", GeneratorInputSequenceLengthHostBuffer->data(), mConfig->beamSize * actualBatchSize * sizeof(int));
    
    return sequenceSampleIdAndLength;
}

void GNMTModel::translate(int actualBatchSize, std::vector<std::pair<int, int>> sequenceSampleIdAndLength, std::shared_ptr<GNMTBindings> bindings, std::shared_ptr<InferenceManager> resources, std::vector<mlperf::ResponseId> rids, bool batchCulling)
{

    auto EncoderBinding = bindings->mEncoderBinding;
    auto GeneratorBinding = bindings->mGeneratorBinding;
    auto ShufflerBinding = bindings->mShufflerBinding;

    auto EncoderInputEmbeddingIndicesHostBuffer = bindings->mExtraResources->mEncoderInputEmbeddingIndicesHostBuffer;
    auto EncoderInputSequenceLengthsHostBuffer = bindings->mExtraResources->mEncoderInputSequenceLengthsHostBuffer;
    auto GeneratorInputSequenceLengthHostBuffer = bindings->mExtraResources->mGeneratorInputSequenceLengthHostBuffer;

    auto BeamSearchInputHostLogProbsCombined = bindings->mExtraResources->mBeamSearchInputHostLogProbsCombined;
    auto BeamSearchInputHostBeamIndices = bindings->mExtraResources->mBeamSearchInputHostBeamIndices;
    auto BeamSearchOutputHostParentLogProbs = bindings->mExtraResources->mBeamSearchOutputHostParentLogProbs;
    auto BeamSearchOutputHostNewCandidateTokens = bindings->mExtraResources->mBeamSearchOutputHostNewCandidateTokens;
    auto BeamSearchOutputHostParentBeamIndices = bindings->mExtraResources->mBeamSearchOutputHostParentBeamIndices;

    auto GeneratorInputLengthPenaltyBuffer = bindings->mExtraResources->mGeneratorInputLengthPenaltyBuffer;
    auto GeneratorCtxByBatchSize = bindings->mGeneratorCtxByBatchSize;

    // Run the encoder
    mEncoder.run(actualBatchSize, EncoderBinding, resources);

    const size_t maxDecoderSeqLen = *std::max_element(EncoderInputSequenceLengthsHostBuffer->data(), EncoderInputSequenceLengthsHostBuffer->data() + actualBatchSize) * 2;

    BeamSearch beamSearch(actualBatchSize, mConfig);

    int generatorBatchSize = actualBatchSize;
    int lastUnfinishedSampleCount = actualBatchSize;
    int unfinishedSampleCount = actualBatchSize;
    // Run Generator for each token
    for (int tok = 0; tok < mConfig->decoderSeqLen; ++tok)
    {
        if (tok > 0)
        {
            // Run the shuffler starting from the second iteration.
            mShuffler.run(generatorBatchSize, ShufflerBinding, resources);
        }

        if (tok == 0)
        {
            // Initialize input buffers for the generator for the 1st iteration
            for(int i = 0; i < mConfig->decoderLayerCount; ++i)
            {
                GeneratorBinding.setHiddInput(i, EncoderBinding.getHiddOutput(i));
                GeneratorBinding.setCellInput(i, EncoderBinding.getCellOutput(i));
            }
        }
        else if (tok == 1)
        {
            // Initialize input buffers for the generator to get data from the shuffler
            GeneratorBinding.setAttentionInput(ShufflerBinding.getAttentionShuffledOutput());
            for(int i = 0; i < mConfig->decoderLayerCount; ++i)
            {
                GeneratorBinding.setHiddInput(i, ShufflerBinding.getHiddShuffledOutput(i));
                GeneratorBinding.setCellInput(i, ShufflerBinding.getCellShuffledOutput(i));
            }
        }
        GeneratorBinding.setLengthPenaltyInput(GeneratorInputLengthPenaltyBuffer->data() + tok);

        // Run the generator
        auto generatorCtx = GeneratorCtxByBatchSize.lower_bound(generatorBatchSize)->second;
        GeneratorBinding.setExecutionContext(generatorCtx);
        mGenerator.run(generatorBatchSize, GeneratorBinding, resources);

        CHECK_EQ(cudaMemcpyAsync(
            BeamSearchInputHostLogProbsCombined->data(),
            GeneratorBinding.getLogProbsCombinedOutput(),
            generatorBatchSize * mConfig->beamSize * sizeof(float),
            cudaMemcpyDeviceToHost, GeneratorBinding.stream()),CUDA_SUCCESS);
        CHECK_EQ(cudaMemcpyAsync(
            BeamSearchInputHostBeamIndices->data(),
            GeneratorBinding.getBeamIndicesOutput(),
            generatorBatchSize * mConfig->beamSize * sizeof(int32_t),
            cudaMemcpyDeviceToHost, GeneratorBinding.stream()),CUDA_SUCCESS);


        if(rids.size() > 0)
        {
            if(unfinishedSampleCount < lastUnfinishedSampleCount)
            {
                // Get the final prediction for the batch.
                int sentBatchSize = lastUnfinishedSampleCount - unfinishedSampleCount;

                vector<vector<int>> finalPredictions(sentBatchSize);
                for (int i = unfinishedSampleCount; i< lastUnfinishedSampleCount; i++)
                {
                    finalPredictions[i - unfinishedSampleCount] = beamSearch.getFinalPrediction(i);
                }

                // Get the German words for the predicted tokens
                std::vector<std::vector<std::string>> tokenWords;
                mIndexer.findWords(tokenWords, finalPredictions, RuntimeInfo::currentBatch);

                std::vector<QuerySampleResponse> response(sentBatchSize);
                std::vector<std::string> responseStrings(sentBatchSize);

                for (int i = 0, j = unfinishedSampleCount; j < lastUnfinishedSampleCount; i++, j++)
                {
                    int sampleId = sequenceSampleIdAndLength[j].first;
                    stringstream translatedText;
                    writeTokenizedSentence(translatedText, tokenWords[i]);

                    responseStrings[i] = translatedText.str();

                    uintptr_t data = reinterpret_cast<mlperf::ResponseId>(responseStrings[i].c_str());
                    response.at(i) = QuerySampleResponse{rids.at(sampleId), data, responseStrings[i].length()};

                }

                QuerySamplesComplete(response.data(), response.size());

                lastUnfinishedSampleCount = unfinishedSampleCount;
            }
        }

        GeneratorBinding.bindingsSynchronize();

        double t0 = seconds();
        unfinishedSampleCount = beamSearch.run(
            generatorBatchSize,
            maxDecoderSeqLen,
            BeamSearchInputHostLogProbsCombined->data(),
            BeamSearchInputHostBeamIndices->data(),
            BeamSearchOutputHostParentLogProbs->data(),
            BeamSearchOutputHostNewCandidateTokens->data(),
            BeamSearchOutputHostParentBeamIndices->data());
        double cpuTime = seconds() - t0;
        mScorerCpuTimeTotal += cpuTime;

        // Check if the beam search finished for all the samples in the batch
        if (unfinishedSampleCount == 0)
            break;
        
        GeneratorBinding.setParentLogProbsInput(bindings->mGeneratorParentLogProbsInput);
        GeneratorBinding.setEmdeddingIndicesInput(bindings->mGeneratorEmdeddingIndicesInput);
        
        GeneratorBinding.setInputCudaBufferFromHost("Scorer_parentLogProbs", BeamSearchOutputHostParentLogProbs->data(), generatorBatchSize * mConfig->beamSize * sizeof(float));
        GeneratorBinding.setInputCudaBufferFromHost("Query Embedding Indices", BeamSearchOutputHostNewCandidateTokens->data(), generatorBatchSize * mConfig->beamSize * sizeof(int32_t));
        ShufflerBinding.setInputCudaBufferFromHost("Scorer_parent_beam_idx", BeamSearchOutputHostParentBeamIndices->data(), generatorBatchSize * mConfig->beamSize * sizeof(int32_t));

        if (batchCulling)
            generatorBatchSize = unfinishedSampleCount;
    }

    if(rids.size() > 0)
    {
        if(unfinishedSampleCount < lastUnfinishedSampleCount)
        {
            // Get the final prediction for the batch.
            int sentBatchSize = lastUnfinishedSampleCount - unfinishedSampleCount;

            vector<vector<int>> finalPredictions(sentBatchSize);
            for (int i = unfinishedSampleCount; i< lastUnfinishedSampleCount; i++)
            {
                finalPredictions[i - unfinishedSampleCount] = beamSearch.getFinalPrediction(i);
            }

            // Get the German words for the predicted tokens
            std::vector<std::vector<std::string>> tokenWords;
            mIndexer.findWords(tokenWords, finalPredictions, RuntimeInfo::currentBatch);

            std::vector<QuerySampleResponse> response(sentBatchSize);
            std::vector<std::string> responseStrings(sentBatchSize);

            for (int i = 0, j = unfinishedSampleCount; j < lastUnfinishedSampleCount; i++, j++)
            {
                int sampleId = sequenceSampleIdAndLength[j].first;
                stringstream translatedText;
                writeTokenizedSentence(translatedText, tokenWords[i]);

                responseStrings[i] = translatedText.str();

                uintptr_t data = reinterpret_cast<mlperf::ResponseId>(responseStrings[i].c_str());
                response.at(i) = QuerySampleResponse{rids.at(sampleId), data, responseStrings[i].length()};

            }

            QuerySamplesComplete(response.data(), response.size());

            lastUnfinishedSampleCount = unfinishedSampleCount;
        }
    }

}
