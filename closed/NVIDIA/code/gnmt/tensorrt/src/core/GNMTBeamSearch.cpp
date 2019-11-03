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

#include "GNMTBeamSearch.h"

std::vector<int> BeamSearch::getFinalPrediction(int sampleId)
{
    assert(!mFinishedCandidates[sampleId].empty());

    // We have a candidate (finished sequence)
    const auto& bestFinishedCandidate = mFinishedCandidates[sampleId].front();
    return backtrack(bestFinishedCandidate.finalTimestepId - 1, sampleId, bestFinishedCandidate.finalRayId);
}

std::vector<int> BeamSearch::backtrack(
    int lastTimestepId,
    int sampleId,
    int lastTimestepRayId) const
{
    std::vector<int> res(lastTimestepId + 1);

    int rayId = lastTimestepRayId;
    for (int timestepId = lastTimestepId; timestepId >= 0; --timestepId)
    {
        const auto& entry = mBeamSearchTable[(timestepId * mBatchSize + sampleId) * mConfig->beamSize + rayId];
        rayId = entry.backtrackId;
        res[timestepId] = entry.vocabularyId;
    }

    return res;
}

bool BeamSearch::hasUnfinishedRays(int sampleId) const
{
    return (static_cast<int>(mFinishedCandidates[sampleId].size()) < mConfig->beamSize);
}

int BeamSearch::getTailWithNoWorkRemaining() const
{
    for (int sampleId = mBatchSize - 1; sampleId >= 0; --sampleId)
    {
        if (hasUnfinishedRays(sampleId))
            return sampleId + 1;
    }
    return 0;
}

int BeamSearch::run(
    int batchSize,
    const size_t maxDecoderSeqLen,
    const float * hostLogProbsCombined,
    const int * hostBeamIndices,
    float * hostParentLogProbs,
    int * hostNewCandidateTokens,
    int * hostParentBeamIndices)
{
    float currentTopKLengthPenaltyMultiplierFinished = mConfig->getLengthPenalyMultiplier(mTimeStep);
    float currentTopKLengthPenaltyMultiplierUnfinished = mConfig->getLengthPenalyMultiplier(mTimeStep + 1);
    float lengthPenaltyMultiplierSavedFinished = mConfig->getLengthPenalyMultiplier(mTimeStep + 1);
    if (mTimeStep == 0)
        mBeamSearchTable.resize(maxDecoderSeqLen * mBatchSize * mConfig->beamSize);
    mTimeStep += 1;
    auto baseBeamSearchTable = mBeamSearchTable.begin() + (mTimeStep - 1) * mBatchSize * mConfig->beamSize;

    for (int sampleId = 0; sampleId < batchSize; ++sampleId)
    {
        // Pointers to the buffers in pinned host memory, which is the ouptut of this CPU part of the beam search
        auto currentNewCandidateTokens = hostNewCandidateTokens + sampleId * mConfig->beamSize;
        auto currentParentBeamIndices = hostParentBeamIndices + sampleId * mConfig->beamSize;
        auto currentParentLogProbs = hostParentLogProbs + sampleId * mConfig->beamSize;

        // Pointer to the part of the backtracking table corresponding to this sample
        auto currentBeamSearchTable = baseBeamSearchTable + sampleId * mConfig->beamSize;

        if (hasUnfinishedRays(sampleId))
        {
            // for each unfinished sample we are iterating through both TopK returned from GPU and the already finished candidates
            // to produce the next beamSize candidates for the next iteration
            // basically, this means that we might end up using only the Top (BeamSize - finishedBeams) from GPU at this point if finished candidates are better than those returned from GPU

            auto& finishedCandidates = mFinishedCandidates[sampleId];

            // We could sort right in the end of the previous timestep (right after finishedCandidates update in the end of the loop),
            // but this would diverge from how TF chooses the best sequence when all candidates sequences become finished in the beam.
            // Note: candidates are sorted from high to low scores. Therefore a lower finishedIndex points to a higher ranked candidate
            std::sort(finishedCandidates.begin(), finishedCandidates.end(),
                [](const FinishedCandidate & a, const FinishedCandidate & b) -> bool { return a.score > b.score; });

            int finishedIndex = 0; // References previously finished candidate which we didn't yet process
            int topKId = 0; // References the single result from TopK which we didn't yet process
            std::vector<FinishedCandidate> newFinishedCandidates; // vector for the new set of finished candddates (might end up having previously finished ones as well)
            for(int dstRayId = 0; dstRayId < mConfig->beamSize; ++dstRayId)
            {
                // dstRayId references ray slot in the beam for the next iteration.
                // These slots are filled with the best candidates from 2 sources: TopK returned from GPU and previously finished candidates

                float topKCombinedLikelihood = hostLogProbsCombined[sampleId * mConfig->beamSize + topKId]; // Logprob from TopK
                int topKVocabularyId = hostBeamIndices[sampleId * mConfig->beamSize + topKId] % mConfig->vocabularySize; // Token ID
                int topKOriginalRayId = hostBeamIndices[sampleId * mConfig->beamSize + topKId] / mConfig->vocabularySize; // Index in the beam, which will be used to gather decoder states at shuffle stage

                // At this stage lengthPenalty shouldn't take EOS into account for the just finished candidate
                float topKCurrentScore = (topKVocabularyId == mConfig->STOP_TOKEN ? currentTopKLengthPenaltyMultiplierFinished : currentTopKLengthPenaltyMultiplierUnfinished) * topKCombinedLikelihood;
                if ((finishedIndex >= static_cast<int>(finishedCandidates.size())) || (topKCurrentScore > finishedCandidates[finishedIndex].score))
                {
                    // This condition gets triggered if the current element from TopK has higher score than the current element from the finished candidate list
                    // OR when there are no more fiinished candidates left
                    // We should use an element from TopK for the entry in the beam search

                    bool justFinished = ((topKVocabularyId == mConfig->STOP_TOKEN) || (mTimeStep >= maxDecoderSeqLen));

                    *(currentNewCandidateTokens + dstRayId) = topKVocabularyId;
                    *(currentParentBeamIndices + dstRayId) = topKOriginalRayId;
                    *(currentParentLogProbs + dstRayId) = justFinished ? mConfig->minimalLogProb : topKCombinedLikelihood; // If the sequence just finished we are not interetsed in this candidate producing TopK results at the next timestep

                    (currentBeamSearchTable + dstRayId)->vocabularyId = topKVocabularyId;
                    (currentBeamSearchTable + dstRayId)->backtrackId = topKOriginalRayId;

                    if (justFinished)
                    {
                        // If TopK candidates just finished, store its information in newFinishedScores
                        // To correlate with TF, instead of just storing optionCurrentScore in newFinishedScores, we need to modify the score. i.e., we now need to take EOS into account
                        float optionNewScore = topKCombinedLikelihood * lengthPenaltyMultiplierSavedFinished;
                        FinishedCandidate newSeq{optionNewScore, static_cast<int>(mTimeStep), dstRayId};
                        newFinishedCandidates.push_back(newSeq);
                    }
                    ++topKId;
                }
                else
                {
                    // Previously finished sequence is better than what we've got with TopK from GPU

                    // We could probably reuse the slot in the beam for some other option, but we are following TF implementation
                    // and just fill the parent loprobab with negative one to avoid any indices from this ray selected at the next timestep
                    *(currentNewCandidateTokens + dstRayId) = mConfig->STOP_TOKEN;
                    *(currentParentBeamIndices + dstRayId) = 0;
                    *(currentParentLogProbs + dstRayId) = mConfig->minimalLogProb;

                    (currentBeamSearchTable + dstRayId)->vocabularyId = mConfig->STOP_TOKEN;
                    (currentBeamSearchTable + dstRayId)->backtrackId = 0;

                    newFinishedCandidates.push_back(finishedCandidates[finishedIndex]);
                    ++finishedIndex;
                }
            }

            // Update the vector of finished candidates for this sample
            finishedCandidates = newFinishedCandidates;
        }
    }

    return (mTimeStep == maxDecoderSeqLen) ? 0 : getTailWithNoWorkRemaining();
}
