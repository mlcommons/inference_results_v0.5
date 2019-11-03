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

#include "BatchSequencer.h"

BatchSequencer::BatchSequencer(const std::map<std::string, std::shared_ptr<BatchStream>>& streams){
    std::vector<std::set<std::string>> suffixSets;

    // Collect all suffixes for all BatchStreams
    for (auto it : streams){
        std::string dirName = it.second->getCalibDir();
        std::string prefix = it.second->getPrefix();

        suffixSets.push_back(getSequenceNumbers(dirName, prefix));
    }


    // Compute the intersection between all suffix sets and store it in lastIntersection
    std::set<std::string> lastIntersection = suffixSets[0];
    std::set<std::string> currIntersection;

    for (auto it : suffixSets){
        std::set_intersection(lastIntersection.begin(), lastIntersection.end(), 
                            it.begin(), it.end(), std::inserter(currIntersection, currIntersection.begin()));

        std::swap (lastIntersection, currIntersection);
        currIntersection.clear();
    }

    // Check whether there are some batches that don't appear in some batchstreams
    for (auto it : suffixSets){
        if (lastIntersection.size() != it.size()){
            std::cout << "Warning, some batchstreams have more batches than others. Omitting sequences that are not available to all." << std::endl;
        }
    }

    // By pushing it in a vector, one can optionally specify a preferred way of sorting
    // Since it doesn't matter for our calibration purposes, we don't do this here
    for (auto it : lastIntersection){
        suffixes.push_back(it);
    }
    
    // Point to the very first suffix
    nextSuffix = suffixes.begin();
}

std::set<std::string> BatchSequencer::getSequenceNumbers(std::string dirName, std::string prefix) {
    std::set<std::string> suffixes;
    auto files = listdir(locateFile(dirName));

    // To avoid ambiguity, we extend the prefix by _b
    // All dumped tensors are store in files of the format 
    // <TensorName>_b<batchNumber>_<timeStep>.dump
    // In case a tensor starts with the same TensorName as another,
    // we would have incorrect results. Ambiguity is still possible 
    // (TensorName containig _b), but is at least limited.
    std::string fullPrefix = prefix + "_b";

    for (auto &fName : files){
        // If the filename starts with inputPrefix, add the suffix to the set
        if (fName.find(fullPrefix) == 0){
            suffixes.insert(fName.substr(prefix.size()));
        }
    }

    return suffixes;
}
