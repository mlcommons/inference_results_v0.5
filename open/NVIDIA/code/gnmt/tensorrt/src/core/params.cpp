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

#include "params.h"



void Params::printHelp(const char *name)
{
    printf("\t-h or --help Print this help message.\n");
    printf("\t--bs <int> Specify the batch size.\n");
    printf("\t--bm <int> Specify the beam width.\n");
    printf("\t--max_persistent_bs <int> Specify the maximum batch size to use persistent lstm.\n");
    printf("\t--num_batches <int> Specify the number of batches (debugging purposes only).\n");
    printf("\t--start_batch <int> Specify the first batch to start processing (debugging purposes only).\n");
    printf("\t--input_file Path to input file. Default file 'newstest2014.tok.bpe.32000.en' is provided with GNMT.\n");
    printf("\t--vocab_file Path to vocabulary file\n");
    printf("\t--output_dir Explicitly provide path to output directory.\n");
    printf("\t-t <str> Specify precision (fp32 or fp16).\n");
    printf("\t--int8Generator Run generator in INT8 precision.\n");
    printf("\t--load_config <str> JSON file where configuration information is stored.\n");
    printf("\t--load_engine <str> Directory where GNMT engine is stored.\n");
    printf("\t--store_engine <str> Directory to store GNMT engine to.\n");
    printf("\t--verbose: verbose output\n");
    printf("\t--validation_dump: Dump intermediate tensors for validation purposes.\n");
    printf("\t--validate <unitName>: Dump intermediate tensors for unit with name unitName. This option can be passed multiple times for as many units as needed (e.g., --validate Encoder --validate Decoder). \n");
    printf("\t--disableBatchCulling: Disable batch culling.\n");
    printf("\t--profile: Enable profiling different components of the benchmark at the cost of reduced overall performance.\n");
    printf("\t--build_only: Build the gnmt engines wihtout running it.\n");
    printf("\t--calibration_phase <int>: Run int8 calibration mode. There are two phases needed: 1 and 2.\n");
    printf("\t--calibration_cache <str> Path to the calibration cache, need for int8Generator mode.\n");
    printf("\t--calibration_data <str> Path to the calibration data. calibration_phase 2 will read from this path.\n");
    printf("\t--seq_len_slots <int>: Specify th enumber of TRT engines to be created for different sequence lengths.\n");
    exit(EXIT_SUCCESS);
}

void Params::parseOptions(int argc, const char **argv)
{
    bool userSpecifiedOutputName = false;

    for (int x = 1; x < argc; ++x)
    {
        std::string tmp(argv[x]);
        if (tmp == "--help")
            printHelp(argv[0]);
        else if (tmp == "-h" && (x + 1) < argc)
            printHelp(argv[0]);

        // Essential configurational parameters
        // Note: these will be ignored if load_engine or load_config are set
        else if (tmp == "--bm" && (x + 1) < argc){
            setBeamSize(atoi(argv[++x]));
        }
        else if (tmp == "-t" && (x + 1) < argc){
            std::string sPrec = argv[++x];
            setPrecision(sPrec);
        }
        else if (tmp == "--int8Generator"){
            setInt8Generator(true);
        }
        else if (tmp == "--vocab_file" && (x + 1) < argc){
            setVocabFile(argv[++x]);
        }
        else if (tmp == "--calibration_phase" && (x + 1) < argc){
            setCalibrationPhase(atoi(argv[++x]));
        }
        else if (tmp == "--calibration_cache" && (x + 1) < argc){
            setCalibrationCache(argv[++x]);
        }
        else if (tmp == "--calibration_data" && (x + 1) < argc){
            setCalibrationData(argv[++x]);
        }

        // IO related parameters
        else if (tmp == "--input_file" && (x + 1) < argc)
            inputFile = argv[++x];
        else if (tmp == "--output_dir" && (x + 1) < argc){
            outDir = argv[++x];
            userSpecifiedOutputName = true;
        }

        // Serialization related parameters
        else if (tmp == "--load_engine" && (x + 1) < argc){
            loadEngine = true;
            if(engineDir.empty()){
                engineDir = argv[++x];
            }
            else{
                std::cout << "Multiple initializations of engine dir: Engine dir was already set to " << engineDir << std::endl;
            }
        }
        else if (tmp == "--store_engine" && (x + 1) < argc){
            storeEngine = true;
            if(engineDir.empty()){
                engineDir = argv[++x];
            }
            else{
                std::cout << "Multiple initializations of engine dir: Engine dir was already set to " << engineDir << std::endl;
            }
        }

        // Debugging related parameters
        else if (tmp == "--num_batches" && (x + 1) < argc)
            nbBatch = atoi(argv[++x]);
        else if (tmp == "--start_batch" && (x + 1) < argc)
            startBatch = atoi(argv[++x]);
        else if (tmp == "--validation_dump"){
            validationDump = true;
        }
        else if (tmp == "--validate" && (x + 1) < argc){
            std::string unitName = argv[++x];
            validateUnits.push_back(unitName);
        }

        // Others
        else if (tmp == "--bs" && (x + 1) < argc){
            batchSize = atoi(argv[++x]);
        }
        else if (tmp == "--max_persistent_bs" && (x + 1) < argc){
            persistentLSTMMaxBatchSize = atoi(argv[++x]);
        }
        else if (tmp == "--load_config" && (x + 1) < argc){
            configFile = argv[++x];
        }
        else if (tmp == "--verbose" )
            G_verbose = true;

        else if (tmp == "--disable_batch_culling"){
            disableBatchCulling = true;
        }
        else if (tmp == "--profile"){
            profile = true;
        }
        else if (tmp == "--build_only"){
            build_only = true;
        }
        else if (tmp == "--seq_len_slots" && (x + 1) < argc)
            seqLenSlots = atoi(argv[++x]);

        else
            printHelp(argv[0]);

    } // end for loop

    // Run smarts on finding the various input files
    if (inputFile.empty()){
        inputFile = locateFile("newstest2014.tok.bpe.32000.en");
    }
    if (vocabFile.empty()){
        vocabFile = locateFile("vocab.bpe.32000.en");
    }

    // In calibration phase 2 we read from calibrationData, fill in default directory
    if (calibrationPhase == 2 && calibrationData.empty()){
        calibrationData = "calib_batches";
    }
    
    // Ensure we can find the file
    if(calibrationPhase == 2){
        calibrationData = locateFile(calibrationData);
    }

    // In calibration phase 2 we write to calibrationCache, in int8Generator, we read from calibrationCache
    // fill in default file name
    if ((calibrationPhase == 2 || enableInt8Generator) && calibrationCache.empty()){
        calibrationCache = "code/gnmt/tensorrt/data/Int8CalibrationCache";
    }

    // Ensure we can find the file, if reading
    if (enableInt8Generator){
        calibrationCache = locateFile(calibrationCache);
    }


    // Create output file name
    if(! userSpecifiedOutputName){

        std::string sPrec = "";
        if(prec == DataType::kHALF){
                sPrec += "_fp16";
        }
        else {
            sPrec += "_fp32";
        }

        if (enableInt8Generator){
            sPrec += "_int8proj";
        }

        outDir = "TRT_GNMT_bs" + std::to_string(batchSize) + "_bm" + std::to_string(beamSize) + sPrec;
    }

    // Create output directory
    bool success = createDirIfNonExistent(outDir);
    if (! success){
        std::cerr << "Could not generate output directory " << outDir << std::endl;
    }

    // Directory to store intermediate tensors
    debugDirName = createUniqueSuffixedName(outDir, createNameWithTimeStamp("gnmt_tensors"));

    std::cout << "Created dir " << debugDirName << std::endl;

    // Filename of final translation
    outFile = outDir + "/translation.txt";
}

bool Params::validateArgs(){
    bool isGood = true;

    bool areWeSerializing = loadEngine || storeEngine;

    // load_config and load_engine are mutually exclusive
    if(loadEngine && !configFile.empty()){
        std::cerr << "Error: Loading engines and config file are mutually exclusive." << std::endl;
        isGood = false;
    }

    if (areWeSerializing && calibrationPhase > 0){
        std::cerr << "Error: Calibration mode cannot be specified when serializing engines." << std::endl;
        isGood = false;
    }

    if (calibrationPhase == 1 && !calibrationData.empty()){
        std::cerr << "Warning: calibration data was specified. "
            << "While in calibration phase 1 we generate data, "
            << "we can only specify its location using --output_dir" << std::endl;
    }

    if (calibrationPhase > 0 && prec != DataType::kFLOAT){
        std::cerr << "Error: when calibrating INT8 scales, precision needs to be FP32" << std::endl;
        isGood = false;
    }

    if (calibrationPhase > 0 && enableInt8Generator){
        std::cerr << "Error: when calibrating INT8 scales, we should not run in INT8 mode just yet." << std::endl;
        isGood = false;
    }

    // Warn if config file will override parameters the user specified.
    if(shouldLoadConfigFromFile() && essentialParamSpecified){
        std::cerr << "Warning: Essential parameters have been specified through command line. "
            << "These will be overriden by the configuration file that was loaded." << std::endl;
    }

    // The following parameters were changed while load_engine was set to True

    return isGood;
}

void Config::initializeFromJSON(std::string jsonFile){
    JSONObject root = getJSONRoot(jsonFile);

    try{
        beamSize = getInt(root, "beamSize");
        enableInt8Generator = getBool(root, "enableInt8Generator");
        prec = strToDataType(getString(root, "precision"));

        // Set and check vocabulary size
        setVocabulary(locateFile(getString(root, "vocabFile")));
        int vocabCheck = getInt(root, "vocabularySize");
        assert(vocabCheck == vocabularySize && "Vocabulary size mismatch");

        int seqLen = getInt(root, "encoderMaxSeqLen");
        int seqLenSlots = getInt(root, "encoderMaxSeqLenSlots");
        setSeqLen(seqLen, seqLenSlots);

        maxBatchSize = getInt(root, "maxBatchSize");
        persistentLSTMMaxBatchSize = getInt(root, "persistentLSTMMaxBatchSize");

        calibrationPhase = getInt(root, "calibrationPhase");
        std::string tmpCalibrationData = getString(root, "calibrationData");
        std::string tmpCalibrationCache = getString(root, "calibrationCache");
        calibrationData = tmpCalibrationData.empty() ? "" : locateFile(tmpCalibrationData);
        calibrationCache = tmpCalibrationCache.empty() ? "" : locateFile(tmpCalibrationCache);

    } catch(JSONParseException& e){
        std::cerr << "Error while parsing " << jsonFile << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

void Config::writeToJSON(std::string jsonFile){

    JSONObject root;

    root[L"beamSize"] = new JSONValue(beamSize);
    root[L"enableInt8Generator"] = new JSONValue(enableInt8Generator);
    root[L"precision"] = new JSONValue(convertToWString(dataTypeToStr(prec)));
    root[L"vocabFile"] = new JSONValue(convertToWString(vocabFile));
    root[L"vocabularySize"] = new JSONValue(vocabularySize);
    root[L"encoderMaxSeqLen"] = new JSONValue(encoderMaxSeqLengths[0]);
    root[L"encoderMaxSeqLenSlots"] = new JSONValue(static_cast<int>(encoderMaxSeqLengths.size()));
    root[L"maxBatchSize"] = new JSONValue(maxBatchSize);
    root[L"persistentLSTMMaxBatchSize"] = new JSONValue(persistentLSTMMaxBatchSize);
    root[L"calibrationPhase"] = new JSONValue(calibrationPhase);
    root[L"calibrationData"] = new JSONValue(convertToWString(calibrationData));
    root[L"calibrationCache"] = new JSONValue(convertToWString(calibrationCache));

    std::wstring outStr = JSON::Stringify(new JSONValue(root));

    wofstream wof(jsonFile);

    wof << outStr << std::endl;

    wof.close();
}

void Config::initializeFromParams(Params params, int seqLen){
    beamSize = params.beamSize;
    setSeqLen(seqLen, params.seqLenSlots);
    prec = params.prec;
    enableInt8Generator = params.enableInt8Generator;

    setVocabulary(params.vocabFile);

    validationDump = params.validationDump;
    validateUnits = params.validateUnits;

    maxBatchSize = params.batchSize;
    persistentLSTMMaxBatchSize = params.persistentLSTMMaxBatchSize;

    calibrationPhase = params.calibrationPhase;
    calibrationData = params.calibrationData;
    calibrationCache = params.calibrationCache;
}

void Config::printConfig(){
    std::cout
        << "Beam Size: " << beamSize << std::endl;
    std::cout << "Max Sequence length:";
    for(auto elem: encoderMaxSeqLengths)
        std::cout << " " << elem;
    std::cout << std::endl;
    std::cout
        << "Precision: " << (isHalf()? "fp16": "fp32") << std::endl
        << "Int8 Generator Enabled: " << enableInt8Generator << std::endl
        << "Vocabulary size: " << vocabularySize << std::endl
        << "Maximum Batch Size: " << maxBatchSize << std::endl;

    if(calibrationPhase != 0){
        std::cout << "Calibration Phase: " << calibrationPhase << std::endl;
    }
}

void Config::addDebugTensor(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name, std::string unitName) const {
    if (! isUnitValidationEnabled(unitName)){
        return;
    }

    addDumpPlugin(network, tensor, name);
}

void Config::addCalibrationDumpTensor(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name) const {
    if(! isCalibrationDumpEnabled()){
        return;
    }

    addDumpPlugin(network, tensor, name);
}

void Config::addDumpPlugin(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name) const {
    ITensor * dumpTensorPluginLayerInputs[] = {*tensor};
    DumpTensorPlugin dumpTensorPlugin(name.c_str(), isHalf());
    auto dumpTensorPluginLayer = network->addPluginV2(dumpTensorPluginLayerInputs, 1, dumpTensorPlugin);
    *tensor = dumpTensorPluginLayer->getOutput(0);
}

