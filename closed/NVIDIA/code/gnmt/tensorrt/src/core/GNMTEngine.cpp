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


#include "GNMTEngine.h"

#include "plugin/dumpTensorPlugin.h"

void GNMTEngine::setup(std::string engineDirName, bool loadEngineFromFile, bool storeEngineToFile)
{
    // Setup engine file name if requested
    std::string engineFile = {};
    if (!engineDirName.empty() ){
        // Create directory if needed
        if (storeEngineToFile && !createDirIfNonExistent(engineDirName)){
            std::cerr << "Could not generate directory " << storeEngineToFile << " to store engine." << std::endl;
            storeEngineToFile = false;
        }
        // Create file name
        engineFile = engineDirName + "/" + mName + ".engine";
    }
    if ((loadEngineFromFile || storeEngineToFile) && engineFile.empty()){
        std::cerr << "No engine file name was specified " << std::endl;
        loadEngineFromFile = false;
        storeEngineToFile = false;
    }

    // Load the engine
    if (loadEngineFromFile){
        loadEngine(engineFile);
    }
    // If no engine file was presented, build the engine from scratch
    else{
        buildEngine();
    }

    // Store the engine if requested
    if(storeEngineToFile){
        storeEngine(engineFile);
    }

    mBindings.resize(mEngines.size());
    for(int i = 0; i < mBindings.size(); ++i)
        mBindings[i].resize(mEngines[i]->getNbBindings());
    allocateOutputBuffers();

    mContexts.resize(mEngines.size());
    for(int i = 0; i < mContexts.size(); ++i)
    {
        mContexts[i] = mEngines[i]->createExecutionContext();
        if (mProfile)
            mContexts[i]->setProfiler(&mProfiler);
    }
}

void GNMTEngine::loadEngine(std::string engineFName){
    std::vector<char> engineData;
    size_t fsize{0};

    // Create runtime
    IRuntime* runtime = createInferRuntime(g2Logger.getTRTLogger());

    mEngines.resize(mConfig->encoderMaxSeqLengths.size());
    for(int i = 0; i < mEngines.size(); ++i)
    {
        int currentSeqLen = mConfig->encoderMaxSeqLengths[i];
        std::stringstream ss;
        ss << engineFName << ".encseqlen_" << currentSeqLen;
        std::string engineFNameWithSeqlen = ss.str();

        // Open engine file
        std::ifstream engineFile(engineFNameWithSeqlen, std::ios::binary);
        if (!engineFile.good())
        {
            std::cerr << "Error loading engine file: " << engineFNameWithSeqlen << std::endl;
            assert(false);
        }

        // Read engine file to memory
        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        engineData.resize(fsize);
        engineFile.read(engineData.data(), fsize);
        engineFile.close();

        // Create engine
        mEngines[i] = UniqueEnginePtr(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    }

    runtime->destroy();
}

void GNMTEngine::storeEngine(std::string engineFName){
    for(int i = 0; i < mEngines.size(); ++i)
    {
        int currentSeqLen = mConfig->encoderMaxSeqLengths[i];
        std::stringstream ss;
        ss << engineFName << ".encseqlen_" << currentSeqLen;
        std::string engineFNameWithSeqlen = ss.str();

        std::ofstream engineFile(engineFNameWithSeqlen, std::ios::binary);
        if (!engineFile)
        {
            std::cerr << "Could not open output engine file: " << engineFNameWithSeqlen << std::endl;
            assert(false);
        }

        IHostMemory* serializedEngine = mEngines[i]->serialize();
        if (serializedEngine == nullptr)
        {
            std::cerr << "Could not serialize engine." << std::endl;
            assert(false);
        }

        engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
        serializedEngine->destroy();
    }
}

void GNMTEngine::buildEngine()
{
    // Build network and builder
    mEngines.resize(mConfig->encoderMaxSeqLengths.size());
    for(int i = 0; i < mEngines.size(); ++i)
    {
        mWeightsManager.clear();
        mProcessedWeightsMap.clear();
        importWeights();

        auto builder = createInferBuilder(g2Logger);
        assert(builder);

        int currentSeqLen = mConfig->encoderMaxSeqLengths[i];

        auto network = builder->createNetwork();
        assert(network);
        configureNetwork(network, currentSeqLen);
        
        // Network alterations for calibration or int8 mode
        if(mConfig->buildCalibrationCache()){
            configureCalibrationCacheMode(builder, network);
        }

        else if (mConfig->enableInt8Generator){
            configureInt8Mode(builder, network);
        }

        // Set paramaters for builder
        builder->setMaxBatchSize(mConfig->maxBatchSize);
        builder->setMaxWorkspaceSize(1UL << 32);
        builder->setHalf2Mode(mConfig->isHalf());

        // build engine
        mEngines[i] = UniqueEnginePtr(builder->buildCudaEngine(*network));
        assert(mEngines[i]);

        // destroy the network and builder
        network->destroy();
        builder->destroy();
    }
}

void GNMTEngine::configureInt8Mode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network)
{
}

void GNMTEngine::configureCalibrationCacheMode(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network)
{
}


void GNMTEngine::run(int batchSize, int encoderSeqLenSlot, cudaStream_t stream)
{
    mContexts[encoderSeqLenSlot]->enqueue(batchSize, &mBindings[encoderSeqLenSlot][0], stream, nullptr);
}

int64_t GNMTEngine::getElemCountPerSample(const char * tensorName) const
{
    int bufferIdx = mEngines[0]->getBindingIndex(tensorName);
    assert(bufferIdx >= 0);
    return samplesCommon::volume(mEngines[0]->getBindingDimensions(bufferIdx));
}

int64_t GNMTEngine::getBufferSizePerSample(const char * tensorName) const
{
    // Different engines have different size requirements for input/output tensor buffers
    // Using the engine with the biggest maxSeqLen - mEngines[0]
    int bufferIdx = mEngines[0]->getBindingIndex(tensorName);
    assert(bufferIdx >= 0);
    return samplesCommon::volume(mEngines[0]->getBindingDimensions(bufferIdx)) * samplesCommon::getElementSize(mEngines[0]->getBindingDataType(bufferIdx));
}

std::shared_ptr<CudaBufferRaw> GNMTEngine::getOutputTensorBuffer(const char * tensorName) const
{
    auto it = mOutBufferMap.find(tensorName);
    assert(it != mOutBufferMap.end());
    return it->second;
}

void GNMTEngine::setInputTensorBuffer(const char * tensorName, const void * data)
{
    for(int i = 0; i < mEngines.size(); ++i)
    {
        int bufferIdx = mEngines[i]->getBindingIndex(tensorName);
        assert(bufferIdx >= 0);
        mBindings[i][bufferIdx] = const_cast<void *>(data);
    }
}

void GNMTEngine::addDebugTensor(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name){
    mConfig->addDebugTensor(network, tensor, name, mName);
}

int GNMTEngine::getBindingIndex(const std::string& tensorName, int encoderSeqLenSlot)
{
    int res = mEngines[encoderSeqLenSlot]->getBindingIndex(tensorName.c_str());
    if (res == -1)
    {
        std::string error = tensorName + " is neither an input nor an ouput of the network.\n";
        throw std::runtime_error(error);
    }
    return res;
}

void GNMTEngine::allocateOutputBuffers()
{
    mOutBufferMap.clear();

    int nbBindings = mEngines[0]->getNbBindings();
    for(int i = 0; i < nbBindings; ++i)
    {
        if (!mEngines[0]->bindingIsInput(i))
        {
            const char * tensorName = mEngines[0]->getBindingName(i);
            int64_t bufferSize = getBufferSizePerSample(tensorName) * mEngines[0]->getMaxBatchSize();
            auto buffer = std::make_shared<CudaBufferRaw>(bufferSize);
            mOutBufferMap.insert(std::make_pair(tensorName, buffer));
            for(int j = 0; j < mEngines.size(); ++j)
            {
                int bufferIdx = mEngines[j]->getBindingIndex(tensorName);
                assert(bufferIdx >= 0);
                mBindings[j][bufferIdx] = buffer->data();
            }
        }
    }
}
