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

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include "common.h"
using namespace nvinfer1;
#include "half.h"
#include "logger.h"
#include <fstream>
#include "CudaBuffer.h"
#include <exception>
#include "JSON.h"   // SimpleJSON from third_party directory


#ifndef __GNMT_UTILS_HPP__
#define __GNMT_UTILS_HPP__


namespace GNMTCoreUtil{

//!
//! \brief Helper function that sorts the batch on sentence length
//!    
//! \param tokenIndices Batch of sentences encoded as a vector of vector of vocabulary indexes
//! \param batchCulling Improve translation speed by reducing the batch size when sentences finish
//!
std::vector<std::pair<int, int>> sortBatch(const std::vector<vector<unsigned int>>& tokenIndices, bool batchCulling);

//!
//! \brief Populate sequence length and indices buffer
//!
//! \param[in] tokenIndices Inputs of token indices
//! \param[in] sequenceSampleIdAndLength Input IDs and sequence lengths
//! \param[out] encoderInputSequenceLengthsHostBuffer Sequence length buffer to be filled based on sequenceSampleIdAndLength
//! \param[out] encoderInputEmbeddingIndicesHostBuffer Embedding Indices buffer to be filled based on sequenceSampleIdAndLength 
//! \param[in] encoderMaxSeqLen Max sequence lengths
//! \param[in] stopToken Stop token to fill encoderInputEmbeddingIndicesHostBuffer
//! 
//!
void calculateSeqLengths(const std::vector<vector<unsigned int>>& tokenIndices, const std::vector<std::pair<int, int>> sequenceSampleIdAndLength, std::shared_ptr<HostBufferInt32> encoderInputSequenceLengthsHostBuffer, 
    std::shared_ptr<HostBufferInt32> encoderInputEmbeddingIndicesHostBuffer, const int encoderMaxSeqLen, const int stopToken);

};



inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//!
//! \brief Get a vector containing all files/diretories in dir
//!
std::vector<std::string> listdir(const std::string& dir);

//!
//! \brief Write a vector of tokenized sentences to an ostream handle
//! \param of ostream handle (E.g., open file handle, or stringstream)
//! \param batch vector of vector of tokenized words. batch[i][j] contains the j'th tokenized subword of sentence i.
//! \param delimiter is the delimiter for bpe-subwords. Subwords end with this delimiter. E.g., "easter" would be "east@@ er".
//! \param convertBPESubWords is a bool that indicates whether we want to merge subwords together.
//!
void writeTokenizedBatch(std::ostream& of, const std::vector<std::vector<std::string>>& batch, std::string delimiter="@@", bool convertBPESubWords=true);

//!
//! \brief Write a vector of toekenized words to an ostream handle
//! \param of ostream handle (E.g., open file handle, or stringstream)
//! \param sentence vector of tokenized words. sentence[j] contains the j'th tokenized subword of sentence.
//! \param delimiter is the delimiter for bpe-subwords. Subwords end with this delimiter. E.g., "easter" would be "east@@ er".
//! \param convertBPESubWords is a bool that indicates whether we want to merge subwords together.
//!
void writeTokenizedSentence(std::ostream& of, const std::vector<std::string>& sentence, std::string delimiter="@@", bool convertBPESubWords=true);

//!
//! \class JSONParseException to be thrown when we encounter parsing exceptions of specific JSON keys
//!
struct JSONParseException: public std::exception{
    JSONParseException(std::string nodeName):
        mNodeName(nodeName){}

    const char * what () const throw () {
        std::string sMsg = (std::string("Could not parse or find field with name \"") + mNodeName + "\"");
        char * cMsg = new char[std::strlen(sMsg.c_str()) + 1];
        std::strcpy(cMsg, sMsg.c_str());
        return cMsg;
    }

    std::string mNodeName;
};


//!
//! \brief Get the root JSON Object from a json file
//! \param fName path to the JSON file
//!
JSONObject getJSONRoot(std::string fName);

//!
//! Some helper functions for extracting values from JSON root objects
//!
int getInt(JSONObject root, const std::string& paramName);
bool getBool(JSONObject root, const std::string& paramName);
std::string getString(JSONObject root, const std::string& paramName);

//!
//! Conversion functions between strings and wide strings
//!
std::wstring convertToWString(const std::string& str1);
std::string convertWideToNarrowString(const std::wstring& wstring);

//!
//! Conversion functions between datatype and their string representation
//!
DataType strToDataType(std::string sPrec);
std::string dataTypeToStr(DataType type);

//!
//! \brief Locates a file from a set of given directories.
//!
//! \param input The name of file we are looking for.
//!
//! \return The path to the file as a string.
//!
inline std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{".",
        "code/gnmt/tensorrt", "code/gnmt/tensorrt/data", "code/gnmt/tensorrt/internal/data",
        "build/preprocessed_data/nmt/GNMT", "build/models/GNMT"};
    return locateFile(input, dirs);
}

//!
//! \brief Get number of lines in a text file
//!
int getNumLinesInTextFile(std::string fname);

//! \brief Create a directory, suffixed by date and hour.
std::string createNameWithTimeStamp(std::string prefix);

// Create a numbered file name starting with fNamePrefix_ and ending with extension, that does not exist yet.
//! \return full path to unique file
//! \warning Not thread safe
std::string createUniqueSuffixedName(std::string dirName, std::string fNamePrefix, std::string extension = "");

//!
//! \brief Check whether the directory exists and create it if necessary.
//!
bool createDirIfNonExistent(std::string dirName);

template <typename T>
void writeToUniqueFile(T* hostPtr, size_t bufferSize, std::string dirName, std::string fNamePrefix){
    bool success = createDirIfNonExistent(dirName);

    if(! success){
        std::cerr << "WARNING: Could not create directory. Skipping." << std::endl;
        return;
    }

    // Create a filename that is suffixed with a unique number.
    std::string fname = createUniqueSuffixedName(dirName, fNamePrefix, ".dump");

    // Write content to file
    std::ofstream ofile(fname);

    for(size_t j = 0; j < bufferSize; j++)
    {
        ofile << std::setprecision(3) << hostPtr[j]  << ", ";
        if(j % 10 == 9)
            ofile << std::endl;
    }

    ofile.close();

}


// In order to remove this class, we would need to either
// 1. Add a currentBatch-field to the Config class, or
// 2. Have all GNMTBase classes keep track to a RuntimeInfo shared_ptr
// This complexity doesn't seem necesary at the moment.
class RuntimeInfo{
public:
    static int currentBatch;
};

// Global variables
extern int SCRATCH;
extern bool G_verbose;


static Logger g2Logger; // Another variable called gLogger is defined somewhere else

void setOutputDataType(nvinfer1::INetworkDefinition* network, DataType precision);

//!
//! \brief Convert a Tensor of FP16 data from GPU to vector of float on CPU
//! \note Space must already be allocated
//! \param src: Source GPU Tensor
//! \param dst: Destination CPU vector (space must already be allocated)
//! \param nbElements: number of elements to copy
//!
void convertGPUHalfToCPUFloat(std::shared_ptr<CudaBufferRaw> src, vector<float>& dst, size_t nbElements);

//!
//! \brief Convert a vector of floats on CPU to a vector of half on CPU
//! \note Space must already be allocated
//! \param src: Source CPU vector
//! \param dst: Destination CPU vector (space must already be allocated)
//! \param nbElements: number of elements to copy
//!
void converCPUFloatToCPUHalf(vector<float>& src, vector<half_float::half>& dst);

using TInfo = std::pair<std::string, Dims3>;

void printDimensions(Dims dims);

class Profiler : public IProfiler
{
public:
    Profiler(int iterations, std::string unitName){
        m_iterations = iterations;
        mUnitName = unitName;
    }

    int m_iterations;
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;
    std::string mUnitName;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
                mProfile.push_back(std::make_pair(layerName, ms));
        else
                record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
                if(G_verbose){
                    printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / m_iterations);
                }
                totalTime += mProfile[i].second;
        }
        printf("Total Time %s (ms): %4.3f\n", mUnitName.c_str(),totalTime / m_iterations);
    }

};

#endif
