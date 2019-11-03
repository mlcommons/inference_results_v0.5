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

#ifndef __GNMT_INDEXER_H__
#define __GNMT_INDEXER_H__

#include "params.h"

#include <unordered_map>

class Indexer
{
public:
    Indexer(std::shared_ptr<Config> config);

    //!
    //! \brief Finds the corresponding indices from the dictionary for a given token word.
    //!
    //! \note Sentences longer than mMaxTokens are clipped.
    //!
    //! \param samples Vector of sentences we want to convert to indices
    //! \param tokenInd Corresponding token indices 
    //!
    void findIndices(const std::vector<std::string>& samples, std::vector<vector<unsigned int>>& tokenInd);

    //!
    //! \brief Finds the corresponding words from the dictionary for the candidate token indices.
    //!
    //! \param tokenWords Corresponding output token words 
    //! \param finalPred Predicted tokens  
    //! \param batch current batch number 
    //!
    void findWords(std::vector<vector<std::string>>& tokenWords, vector<vector<int>>& finalPred, unsigned int batch);

private:
    std::shared_ptr<Config> mConfig;
    std::string mVocabFile;
    std::vector<std::string> dictWords;
    std::unordered_map<std::string, int> dictMap;
    int mMaxTokens;
};

#endif
