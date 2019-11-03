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

#ifndef __GNMT_CORE_H__
#define __GNMT_CORE_H__

#include <vector>
#include <string>
#include "GNMTEncoder.h"
#include "GNMTShuffler.h"
#include "GNMTIndexer.h"
#include "params.h"
#include "GNMTGenerator.h"

using CudaBufferMap = std::map<std::string, std::shared_ptr<CudaBufferRaw>>;

class GNMTCore{
public:

	//!
	//! \brief Initializes all GNMTBase classes that compose GNMT
	//! \note Buffers are not initialized here
	//!
	GNMTCore(std::shared_ptr<Config> config, bool loadEngine=false, bool storeEngine=false, std::string engineDir="", bool profile=false):
		mConfig(config),
		mLoadEngine(loadEngine),
		mStoreEngine(storeEngine),
		mEngineDir(engineDir),
		mIndexer(config),
		mEncoder(config, profile),
		mShuffler(config, profile),
		mGenerator(config, profile),
		mProfile(profile) {;}

	//!
	//! \brief Create and connect engines for GNMT network
	//!
	//! \post All buffers are setup here:
	//! * mEncoderBuffers;
	//! * mQueryBuffers;
	//! * mAttentionBuffers;
	//! * mDecoderBuffers;
	//! * mShufflerOutputBuffers;
	//! * mShufflerInputBuffers;	
	//! \post mScorerCpuTimeTotal is reset to 0
	//! \post if mStoreEngine, a json file with data from mConfig gets dumped to mEngineDir.
	//!
	void setup();


	//!
	//! \brief Build the mGenerator engine and run it through calibration
	//!
	void buildCalibrationCache();

	//!
	//! \brief Translate input text and return output text
	//! \param batch A batch of samples in a source language (e.g., English)
	//! \param batchCulling Improve translation speed by reducing the batch size when sentences finish
	//! \return The tokenized translation of batch in a destination language (e.g., German). 
	//!
	//! \pre The batch needs to contain tokenized- and subword splitted- version of the sentences.
	//!
	//! \note That the input is a vector of sentences, whereas the output is a vector of vectors of (sub-)words.
	std::vector <std::vector<std::string>> translate(std::vector<std::string> batch, bool batchCulling);

	//!
	//! \brief Report the aggregated runtimes on a per-engine basis
	//!
	void reportAggregateEngineTime();
	
private:
	//!
	//! \brief Helper function that translates a token-vector
	//! \param[in] tokenIndices Batch of sentences encoded as a vector of vector of vocabulary indexes
	//! \param[out] tokenWords Translated sentences, in the form of a vector of vector of subword-strings
	//! \param[in] batchCulling Improve translation speed by reducing the batch size when sentences finish
	//!
	void translate(const std::vector<vector<unsigned int>>& tokenIndices, std::vector<vector<std::string>>& tokenWords, bool batchCulling);

	//!
	//! Pointer to the configuration of GNMT
	//!
	std::shared_ptr<Config> mConfig;

	//!
	//! Serialization related data
	//!
	bool mLoadEngine;
	bool mStoreEngine;
	std::string mEngineDir;

	//!
	//! Engines that compes GNMT
	//!
	Indexer mIndexer;
	Encoder mEncoder;
	Shuffler mShuffler;
	Generator mGenerator;

	//!
	//! Buffers that contain input data for the engines
	//!
	std::shared_ptr<CudaBufferInt32> mEncoderInputEmbeddingIndicesBuffer;
	std::shared_ptr<HostBufferInt32> mEncoderInputEmbeddingIndicesHostBuffer;

	std::shared_ptr<CudaBufferInt32> mEncoderInputSequenceLengthsBuffer;
	std::shared_ptr<HostBufferInt32> mEncoderInputSequenceLengthsHostBuffer;

	std::shared_ptr<CudaBufferInt32> mGeneratorInitialCandidateTokensBuffer;
	std::shared_ptr<CudaBufferRaw> mGeneratorInitialAttentionBuffer;
	std::shared_ptr<CudaBufferFP32> mGeneratorInitialParentLogProbsBuffer;

	std::shared_ptr<CudaBufferInt32> mGeneratorInputSequenceLengthBuffer;
	std::shared_ptr<HostBufferInt32> mGeneratorInputSequenceLengthHostBuffer;

	std::shared_ptr<CudaBufferInt32> mGeneratorInputIndicesBuffer;
	std::shared_ptr<CudaBufferFP32> mGeneratorInputParentCombinedLikelihoodsBuffer;
	std::shared_ptr<CudaBufferFP32> mGeneratorInputLengthPenaltyBuffer;

	std::shared_ptr<CudaBufferInt32> mShufflerInputParentBeamIndicesBuffer;

	// Host buffers for the BeamSearch component running on CPU
    // Input for the CPU part
    std::shared_ptr<HostBufferFP32> mBeamSearchInputHostLogProbsCombined;
    std::shared_ptr<HostBufferInt32> mBeamSearchInputHostBeamIndices;
    // Output of the CPU part
    std::shared_ptr<HostBufferFP32>  mBeamSearchOutputHostParentLogProbs;
    std::shared_ptr<HostBufferInt32> mBeamSearchOutputHostNewCandidateTokens;
    std::shared_ptr<HostBufferInt32> mBeamSearchOutputHostParentBeamIndices;

	//! Runtime of the CPU part of the scorer
	double mScorerCpuTimeTotal = 0.0;
	bool mProfile;

	//! Filename of configuration file in case we will serialize it
	const std::string mConfigFileName {"config.json"};
};


#endif
