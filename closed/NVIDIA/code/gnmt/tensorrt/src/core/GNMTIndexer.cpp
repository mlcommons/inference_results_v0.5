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


#include "GNMTIndexer.h"

Indexer::Indexer(std::shared_ptr<Config> config):
        mConfig(config),
        mVocabFile(config->vocabFile),
        mMaxTokens(config->getMaxEncoderSeqLen())
{

    std::string line;
    ifstream vocabFile(mVocabFile);

    if (!vocabFile)
    {
        cout << "Error opening vocabulary file" << mVocabFile << endl;
        exit(EXIT_FAILURE);
    }

    while (std::getline(vocabFile, line))
    {
        dictMap.insert({line, static_cast<int>(dictWords.size())});
        dictWords.push_back(line);
    }
}

void Indexer::findIndices(const std::vector<std::string>& samples, std::vector<vector<unsigned int>>& tokenInd)
{  
    // Check whether there is enough space to store all sentences
    assert(tokenInd.size() >= samples.size());

    std::fill(tokenInd.begin(), tokenInd.end(), std::vector<unsigned int>());

    for (size_t i = 0; i < samples.size(); ++i)
    {

        istringstream sentence(samples[i]);
        while (!sentence.eof() && tokenInd[i].size() < mMaxTokens) {

            std::string token;
            sentence >> token;
            
            int orig_index = -1;
            auto it = dictMap.find(token);
            if(it != dictMap.end())
                orig_index = it->second;
            
            unsigned int enc_index;

            // If sentence ends with empty string, we should stop inserting indices
            if (token == "" && (sentence.tellg() == -1)){
                // Inserting a STOP_TOKEN here increases sequence length by 1 and affects the final output
                break;
            }
            // Encode out-of-vocabulary symbols
            else if (orig_index < 0){
                enc_index = mConfig->UNK_TOKEN;
            }
            // First couple indices are reserved
            else{
                enc_index = orig_index + mConfig->RESERVED_WORDS;
            }
            
            tokenInd[i].push_back(enc_index);
            
        }

    }
}

void Indexer::findWords(std::vector<vector<std::string>>& tokenWords, vector<vector<int>>& finalPred, unsigned int batch)
{
    for (unsigned int i = 0; i < finalPred.size(); ++i)
    {
        tokenWords.push_back(std::vector<std::string>());
        for (unsigned int j = 0; j < finalPred[i].size(); ++j) {
            if (finalPred[i][j] - mConfig->RESERVED_WORDS >= 0)
                tokenWords[i].push_back(dictWords[finalPred[i][j] - mConfig->RESERVED_WORDS]);
            else if (finalPred[i][j] == mConfig->UNK_TOKEN){
                tokenWords[i].push_back(mConfig->UNK);
            }
        }
        // Drop incomplete word in the end of the sequence
        for(int j = tokenWords[i].size() - 1; j > 0; --j) {
            if ((tokenWords[i][j].size() >= 2) && (*(tokenWords[i][j].end() - 1) == '@') && (*(tokenWords[i][j].end() - 2) == '@'))
                tokenWords[i].resize(tokenWords[i].size() - 1);
            else
                break;
        }
    }
}
