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

#ifndef GNMT_BEAM_SEARCH_H
#define GNMT_BEAM_SEARCH_H

#include "params.h"

//!
//! \brief The CPU data for a batch. The user should allocate it and associate it with a specific batch.
//!
class BeamSearch
{
    friend class Scorer;
     
    struct Ray
    {
        int vocabularyId;
        int backtrackId;
    };

    //!
    //! \brief The structure contains info about the already finished sequence
    //!
    struct FinishedCandidate
    {
        float score; // to compare to new candidates as we go along with timesteps 
        int finalTimestepId; // The length of the sequence, with EOS counted
        int finalRayId; // Where to start backtracking
    };

public:
    //!
    //! \brief Run the CPU part of the Scorer
    //!
    //! \param[in]      batchSize Current batch size, could be smaller than the original one due to batch culling  
    //! \param[in, out] scorerData The scorer data for the batch.
    //! \param[in]      maxDecoderSeqLen The limit of the output sequence length.
    //!
    int run(
        int batchSize,
        const size_t maxDecoderSeqLen,
        const float * hostLogProbsCombined,
        const int * hostBeamIndices,
        float * hostParentLogProbs,
        int * hostNewCandidateTokens,
        int * hostParentBeamIndices);

    //!
    //! \brief Get the final prediction of a sample.
    //!
    //! \param scorerData Sample ID.
    //!
    //! \return The predicted token indices for the specified sequence in the batch.
    //!
    std::vector<int> getFinalPrediction(int sampleId);

    //!
    //! \brief Constructor of ScorerData.
    //! \param batchSize The batch size.
    //!
    BeamSearch(size_t batchSize, std::shared_ptr<Config> config)
        : mBatchSize(batchSize)
        , mConfig(config)
        , mTimeStep(0)
    {
        mFinishedCandidates.resize(mBatchSize);
    };

    int getTailWithNoWorkRemaining() const;

protected:
    std::vector<int> backtrack(
        int lastTimestepId,
        int sampleId,
        int lastTimestepRayId) const;

    bool hasUnfinishedRays(int sampleId) const;

protected:
    int mBatchSize;
    std::shared_ptr<Config> mConfig;
    size_t mTimeStep;                      //!< Current time step.
    std::vector<Ray> mBeamSearchTable; // Dynamically resized vector to allow final backtracking 
    std::vector<std::vector<FinishedCandidate>> mFinishedCandidates; // Each element is the sorted vector of finished sequences, up to beamSize elements for each sample
};

#endif
