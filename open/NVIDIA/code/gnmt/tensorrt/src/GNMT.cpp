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

#include <fstream>
#include <iostream>

#include "GNMTCore.h"
#include "GNMTIndexer.h"
#include "utils.h"
#include "params.h"
#include "plugin/dumpTensorPlugin.h"


int main(int argc, const char** argv)
{

    Params params;

    params.parseOptions(argc, argv);
    if (!params.validateArgs()){
        std::cerr << "Parameter check failed. Exiting" << std::endl;
        exit(-1);
    }

    std::shared_ptr<Config> config(nullptr);

    if(params.shouldLoadConfigFromFile()){
        config.reset(new Config(params.getConfigFile()));
    }
    else{
        config.reset(new Config(params));
    }

    DumpTensorPlugin::setDirName(params.debugDirName.c_str());

    // Read input text
    std::string line;
    ifstream inputEnFile(params.inputFile);
    if (!inputEnFile)
    {
        cout << "Error opening input file " << params.inputFile << endl;
        exit(EXIT_FAILURE);
    }
    std::vector<std::string> enSentence;    //! Vector that contains entire corpus text
    while (std::getline(inputEnFile, line))
    {
        enSentence.push_back(line);
    }

    // Calculate batch related parameters
    int maxBatches = ceil((float) enSentence.size() / params.batchSize);

    // Calculate number of batches and last batch number
    if (params.nbBatch < 0){
        params.nbBatch = maxBatches;
    }
    params.nbBatch = min(maxBatches - params.startBatch, params.nbBatch);
    params.stopBatch = params.startBatch + params.nbBatch;

    // Config for engine
    config->printConfig();

    // Build GNMTCore
    GNMTCore gnmt(config, params.loadEngine, params.storeEngine, params.engineDir, params.profile);

    // In calibrationPhase 2, we only want to build the calibration cache, and don't want to run the main code path
    if (params.calibrationPhase == 2){
        gnmt.buildCalibrationCache();
        return 0;
    }
    
    // Processing parameters
    cout << "Actual Batch Size: " << params.batchSize << ", num  batches: " << params.nbBatch
        << " (start @ " << params.startBatch << " end @ " << params.stopBatch - 1 << ")"
        << ", Input file: " << params.inputFile << ", Output file: " << params.outFile << endl;

    // Setup GNMT
    gnmt.setup();

    if (params.build_only)
        return 0;


    // Process input document batch by batch
    ofstream of(params.outFile);
    stringstream translatedText;

    double t0 = seconds();

    std::vector <std::string> currentSamples (params.batchSize);

    for (RuntimeInfo::currentBatch = params.startBatch; RuntimeInfo::currentBatch < params.stopBatch; RuntimeInfo::currentBatch++)
    {
        // Prepare the batch
        auto startSample = RuntimeInfo::currentBatch * params.batchSize;
        auto endSample = min( (RuntimeInfo::currentBatch+1) * params.batchSize, (int) enSentence.size());
        int actualBatchSize = endSample-startSample;

        // TBD: USE POINTER ARITHMETIC HERE to avoid copying (use vector<string*> enSentence and vector<string*> currentSamples)
        std::copy(&enSentence[startSample], &enSentence[endSample], currentSamples.begin());
        currentSamples.resize(actualBatchSize);   //! Shrink if needed.

        std::vector<std::vector<std::string>> tokenWords = gnmt.translate(currentSamples, !params.disableBatchCulling);

        // Write predicted German sentences to output file
        writeTokenizedBatch(translatedText, tokenWords);
    }

    of << translatedText.str();

    double translation_time = seconds() - t0;
    double translation_speed = enSentence.size()/translation_time;
    printf("\nGNMT takes %.3fs \n", translation_time);
    printf("\nTranslation speed is %.2f sentences/sec\n", translation_speed);

    gnmt.reportAggregateEngineTime();

    return 0;
}
