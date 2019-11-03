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

#include "utils.h"
#include <string>
#include <iostream>
#include <dirent.h> // for listdir

int SCRATCH = 28;
bool G_verbose = false;

int RuntimeInfo::currentBatch = -1;

std::vector<std::string> listdir(const std::string& dir) {
    std::vector<std::string> files;
    std::shared_ptr<DIR> dirPtr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *direntPtr;
    if (!dirPtr) {
        std::cout << "Error opening : " << std::strerror(errno) << dir << std::endl;
        return files;
    }

    while ((direntPtr = readdir(dirPtr.get())) != nullptr) {
        files.push_back(std::string(direntPtr->d_name));
    }
    return files;
}

std::vector<std::pair<int, int>> GNMTCoreUtil::sortBatch(const std::vector<vector<unsigned int>>& tokenIndices, bool batchCulling)
{
    int actualBatchSize = tokenIndices.size();
    std::vector<std::pair<int, int>> sequenceSampleIdAndLength(actualBatchSize);
    for (int sampleId = 0; sampleId < actualBatchSize; ++sampleId)
        sequenceSampleIdAndLength[sampleId] = std::make_pair(sampleId, static_cast<int>(tokenIndices[sampleId].size()));
    // If batch culling is on we sort samples inside the batch relying on a reasonable heuristics:
    // Shorter sequences tend to produce shorter translations
    // This will allow to shring batch efficiently while generating output sequences
    if (batchCulling)
    {
        std::sort(
            sequenceSampleIdAndLength.begin(),
            sequenceSampleIdAndLength.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });
    }
    return sequenceSampleIdAndLength;
}

void GNMTCoreUtil::calculateSeqLengths(const std::vector<vector<unsigned int>>& tokenIndices, const std::vector<std::pair<int, int>> sequenceSampleIdAndLength, std::shared_ptr<HostBufferInt32> encoderInputSequenceLengthsHostBuffer, 
    std::shared_ptr<HostBufferInt32> encoderInputEmbeddingIndicesHostBuffer, const int encoderMaxSeqLen, const int stopToken)
{
    int actualBatchSize = tokenIndices.size();
    for (int i = 0; i < actualBatchSize; i++) {
        int sequenceLength = sequenceSampleIdAndLength[i].second;
        assert(sequenceLength <= encoderMaxSeqLen);
        encoderInputSequenceLengthsHostBuffer->data()[i] = sequenceLength;
        int sampleId = sequenceSampleIdAndLength[i].first;
        std::copy_n(tokenIndices[sampleId].begin(), sequenceLength, encoderInputEmbeddingIndicesHostBuffer->data() + encoderMaxSeqLen * i);
        // Fill the remaining tokens with STOP
        std::fill_n(encoderInputEmbeddingIndicesHostBuffer->data() + encoderMaxSeqLen * i + sequenceLength, encoderMaxSeqLen - sequenceLength, stopToken);
    }
}

void writeTokenizedBatch(std::ostream& of, const std::vector<std::vector<std::string>>& batch, std::string delimiter, bool convertBPESubWords){
    for (unsigned int i = 0; i < batch.size(); i++)
    {
        writeTokenizedSentence(of, batch[i]);
        of << endl;
    }
}

void writeTokenizedSentence(std::ostream& of, const std::vector<std::string>& sentence, std::string delimiter, bool convertBPESubWords){
    size_t delimiterLen = delimiter.size();
    bool glueSubwords = (delimiter != "") && convertBPESubWords;

    for (unsigned int j = 0; j < sentence.size(); ++j)
    {
        size_t wordLen = sentence[j].size();
        if (glueSubwords && wordLen >= delimiterLen && sentence[j].find(delimiter, wordLen-delimiterLen) != std::string::npos){
            of << sentence[j].substr(0, wordLen-delimiterLen);
        }
        else if (j == sentence.size() - 1) {
            of << sentence[j];
        }
        else
        {
            of << sentence[j] << " ";
        }
    }
}

JSONObject getJSONRoot(std::string fName){
    // Read the JSON file
    wifstream in(fName.c_str());
    if (in.is_open() == false)
        std::cerr << "Couldn't open config file " << fName << std::endl;

    wstring line;
    wstring data = L"";
    while (getline(in, line))
    {
        data += line;
        if (!in.eof()) data += L"\n";
    }

    // Parse the data
    JSONValue *value = JSON::Parse(data.c_str());

    // Did it go wrong?
    if (value == NULL || value->IsObject() == false)
    {
        std::cerr << "Unable to parse " << fName << std::endl;
        assert(false);
    }
   
    // Retrieve the main object
    return value->AsObject();
}

std::wstring convertToWString(const std::string& str1){
    std::wstring str2(str1.length(), L' ');
    std::copy(str1.begin(), str1.end(), str2.begin());
    return str2;
}

std::string convertWideToNarrowString(const std::wstring& str1){
    std::string str2(str1.length(), ' ');
    std::copy(str1.begin(), str1.end(), str2.begin());
    return str2;
}

int getInt(JSONObject root, const std::string& paramName){
    std::wstring wsParamName = convertToWString(paramName);
    const wchar_t* wParamName = wsParamName.c_str();

    if (root.find(wParamName) == root.end() || !root[wParamName]->IsNumber()){
        throw JSONParseException(paramName);
    }

    double dNum = root[wParamName]->AsNumber();

    return (int) dNum;
}

bool getBool(JSONObject root, const std::string& paramName){
    std::wstring wsParamName = convertToWString(paramName);
    const wchar_t* wParamName = wsParamName.c_str();

    if (root.find(wParamName) == root.end() || !root[wParamName]->IsBool()){
        throw JSONParseException(paramName);
    }

    return root[wParamName]->AsBool();
}

std::string getString(JSONObject root, const std::string& paramName){
    std::wstring wsParamName = convertToWString(paramName);
    const wchar_t* wParamName = wsParamName.c_str();

    if (root.find(wParamName) == root.end() || !root[wParamName]->IsString() ){
        throw JSONParseException(paramName);
    }

    return convertWideToNarrowString(root[wParamName]->AsString());
}

DataType strToDataType(std::string sPrec){
    if(sPrec == "fp32"){
        return DataType::kFLOAT;   
    }
    else if (sPrec == "fp16"){
        return DataType::kHALF;
    }
    else{
        std::cout << "Could not recognize precision setting. Assuming fp32..." << sPrec << std::endl;
        return DataType::kFLOAT;
    }
}

std::string dataTypeToStr(DataType type){
    if(type == DataType::kFLOAT){
        return "fp32";
    }
    else if(type == DataType::kHALF){
        return "fp16";
    }
    else{
        return "unknown datatype";
    }
}

bool createDirIfNonExistent(std::string dirName){
    struct stat info;

    bool dir_exists = stat(dirName.c_str(), &info) == 0;
    if ( ! dir_exists ) {
        dir_exists = (mkdir(dirName.c_str(), 0777) == 0);
    }
    return dir_exists;
}

int getNumLinesInTextFile(std::string fname){
    int count = 0;
    string line;

    fstream file(fname, ios_base::in);

    if (!file.good()) {
	throw std::runtime_error("getNumLinesInTextFile: error when opening file " + fname);
    }

    while (getline(file, line))
        count++;

    return count;
}

void convertGPUHalfToCPUFloat(std::shared_ptr<CudaBufferRaw> src, vector<float>& dst, size_t nbElements){
    // Check if there is sufficient space allocated
    assert(dst.size() == nbElements);

    vector<half_float::half> fp16_tensor(nbElements);
    cudaMemcpy(fp16_tensor.data(), src->data(), nbElements * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < fp16_tensor.size(); i++)
    {
        // Note that operator float is overloaded in the half_float::half class
        // TBD: this could be parallelized or perhaps offloaded to GPU
        dst[i] = float(fp16_tensor[i]);
    }
}

void converCPUFloatToCPUHalf(vector<float>& src, vector<half_float::half>& dst){
    assert(src.size() == dst.size());
    
    for (unsigned int i = 0; i < src.size(); i++)
    {
        dst[i] = half_float::half(src[i]);
    }
}

std::string createNameWithTimeStamp(std::string prefix){
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::string path = "./" + prefix + "_";
    std::stringstream tmp_os; 
    tmp_os << std::put_time(&tm, "%m%d%y_%H");
    path = path + tmp_os.str();

    return path;
}

std::string createUniqueSuffixedName(std::string dirName, std::string fNamePrefix, std::string extension){
    struct stat info;

    int i = 0;
    std::string fname;
    do {
        fname = dirName + "/" + fNamePrefix + "_" + std::to_string(i) + extension;
        ++i;
    }while(stat(fname.c_str(), &info) == 0);

    return fname;
}

/*
 * brief: set all outputs to the given precision
 * note: only use this if you are sure all outputs need to be set to the same precision
 */
void setOutputDataType(nvinfer1::INetworkDefinition* network, DataType precision){
    for(int i = 0; i < network->getNbOutputs(); i++){
        auto out_T = network->getOutput(i);
        out_T->setType(precision);
    }
}
    
void printDimensions(Dims dims){
    std::cout << "Dimensions: ";
    for (int i = 0 ; i < dims.nbDims; i++){std::cout << dims.d[i] << ", "; }
    std::cout << std::endl;
}

