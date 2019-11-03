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

#include <string>
#include "BatchStream.h"
#include <vector>
#include <map>
#include <set>

//!
//! @class BatchSequencer
//!
//! @brief Helper class to point to the next batch sequence
//!
//! @description This class looks at all the input tensors use for the calibrator,
//! and finds the common filename suffixes.
//! By using this class, we can use the same suffix for all tensors at the same time
//!
class BatchSequencer{
public:
    BatchSequencer(const std::map<std::string, std::shared_ptr<BatchStream>>& streams);

    //!
    //! @brief Get the next suffix
    //! @post Advances the suffix-pointer by 1
    //! 
    std::string getNextSuffix(){
        if(! hasNextSuffix()){
            std::cerr << "Warning: no more batches found." << std::endl;
            assert(false);
        }
        return *(nextSuffix++);
    }

    //!
    //! @brief Are there more suffixes available or did we use them all?
    //!
    bool hasNextSuffix() const{
        return (nextSuffix != suffixes.end());
    }


    //!
    //! @brief What's the total number of suffixes available?
    //!
    size_t getNumSequences() const{
        return suffixes.size();
    }

private:
    //!
    //! @brief Helper function to get the suffixes of files that start with a specific prefix.
    //!
    std::set<std::string> getSequenceNumbers(std::string dirName, std::string prefix);

    std::vector<std::string> suffixes;
    std::vector<std::string>::iterator nextSuffix;
};
