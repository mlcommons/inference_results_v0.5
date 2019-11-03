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

#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <string>
#include <vector>
#include "utils.h"
#include "NvInfer.h"
#include "JSON.h"       // SimpleJSON from third_party directory
#include "plugin/dumpTensorPlugin.h"  // DumpTensorPlugin


//!
//! \class Params is used to process and validate command line parameters
//!
class Params{

public:
    //!
    //! Changing essential parameters should set this flag
    //! as it will be used during validateArgs
    //!
    bool essentialParamSpecified {false};

    //! Essential parameters
    int beamSize {10};
    DataType prec {DataType::kFLOAT};
    bool enableInt8Generator {false};
    std::string vocabFile {};   //! Initialize empty to avoid not finding file
    int calibrationPhase {0};
    std::string calibrationData {};
    std::string calibrationCache {};

    //! Parameters for serialization
    std::string engineDir {};
    bool loadEngine {false};
    bool storeEngine {false};

    //! Debug parameters
    int nbBatch {-1};   //! Negative number means process all batches
    int startBatch {0};
    int stopBatch {0};  //! Will be calculated later
    bool validationDump {false};
    std::vector<std::string> validateUnits {};
    std::string debugDirName {createNameWithTimeStamp("gnmt_tensors")};
    std::string outDir {"./"};

    //! JSON file that contains parameters
    //! \note: any "essential parameters" will be overriden
    std::string configFile {};

    std::string inputFile {};   //! Initialize empty to avoid not finding file
    std::string outFile {"TRT_GNMT.txt"};

    int batchSize {5};  //! Batch size we will use to process our input text
    int persistentLSTMMaxBatchSize {8}; //! Maximum batch size to use persistent lstm for encoder

    bool disableBatchCulling {false};  // Apply dynamic batch reduction as samples run out of unfinished candidates
    bool profile {false};              // Enable profiling different components of the benchmark at the cost of reduced overall performance
    bool build_only {false};              // Build the gnmt engines wihtout running it

    // Defines how many TRT engines should be created for different maxSeqLen
    int seqLenSlots {1};

public:
    bool shouldLoadConfigFromFile(){
        return !configFile.empty() || loadEngine;
    }
    std::string getConfigFile(){
        if (!configFile.empty()){
            return configFile;
        }
        else{
            return engineDir + "/config.json";
        }
    }

    //!
    //! \brief Print help message to instruct on command line options
    //!
    void printHelp(const char *name);

    //!
    //! \brief parse command line options and populate this object's member variables accordingly
    //!
    void parseOptions(int argc, const char **argv);

    //!
    //! \brief Check whether the specified parameters are consistent
    //!
    bool validateArgs();

    //!
    //! \brief The following functions ensure that we flag any request to change essential parameters
    //! \note This is necessary for consistency checks in validateArgs()
    //!
    void setBeamSize(int beamSize_){
        beamSize = beamSize_;
        essentialParamSpecified = true;
    }

    void setPrecision(const std::string& sPrec){
        prec = strToDataType(sPrec);
        essentialParamSpecified = true;
    }

    void setInt8Generator(bool enable){
        enableInt8Generator = enable;
        essentialParamSpecified = true;
    }

    void setVocabFile(std::string vocabFile_){
        vocabFile = vocabFile_;
        essentialParamSpecified = true;
    }

    //!
    //! \note This method should at most set heuristics (e.g., best batch size)
    //! flags or options to ensure correct working should be handled from within the config class
    //!
    void setCalibrationPhase(int phase_no){
        calibrationPhase = phase_no;

        if (calibrationPhase == 1){
            batchSize = 1;
        }

        else if (calibrationPhase == 2){
            batchSize = 1;
        }

        else{
            std::cerr << "Warning, unknown calibration phase, please select phase 1 or 2" << std::endl;
            assert(false);
        }

        essentialParamSpecified = true;
    }

    void setCalibrationData(std::string path){
        calibrationData = path;
        essentialParamSpecified = true;
    }

    void setCalibrationCache(std::string path){
        calibrationCache = path;
        essentialParamSpecified = true;
    }

};

class Config{
public:
    //!
    //! Use a params object to initialize
    //!
    Config(Params params){
        initializeFromParams(params);
    }

    //!
    //! Use a JSON file to initialize
    //! \note the following variables remain default initialized:
    //! * debugDirName
    //! * validationDump
    //! * validateUnits
    //!
    Config(std::string jsonFile){
        initializeFromJSON(jsonFile);
    }

    //!
    //! \brief Set sequence lengths for encoder and decoder
    //! \param encoderSeqlength Maximum sequence length for the encoder
    //!
    void setSeqLen(int encoderSeqLen, int seqLenSlots){
        encoderMaxSeqLengths.resize(seqLenSlots);
        for(int i = encoderMaxSeqLengths.size(); i > 0; --i)
            encoderMaxSeqLengths[encoderMaxSeqLengths.size() - i] = encoderSeqLen * i / encoderMaxSeqLengths.size();
        // Setting decoderSeqLen to the maximum possible value
        decoderSeqLen = 2 * encoderMaxSeqLengths[0];
    }

    int getBestEncoderSeqLenSlot(int encoderSeqLen) {
        for(int i = encoderMaxSeqLengths.size() - 1; i >= 0; --i) {
            if (encoderSeqLen <= encoderMaxSeqLengths[i])
                return i;
        }
        return 0;
    }

    int getMaxEncoderSeqLen(){
        return encoderMaxSeqLengths[0];
    }

    void writeToJSON(std::string jsonFile);

    void initializeFromJSON(std::string jsonFile);

    void initializeFromParams(Params params, int seqLen=128);

    bool isHalf() const {
        return prec == DataType::kHALF;
    }

    // TBD: Idealy, this is how we should refactor:
    // GNMTBase classes will have an mValidationEnabled flag
    // In addition they will keep a list of serialized Tensors.
    // Since this is too much work we just disable validation and (de-)serialization combinations for now
    bool isUnitValidationEnabled(std::string unit_name) const{
        if(validationDump){
            return true;
        }
        return (std::find(validateUnits.begin(), validateUnits.end(), unit_name) != validateUnits.end());
    }


    //!
    //! \note To avoid serializing too many variables along with their dependencies,
    //! we are using a number of getter functions instead
    //!

    //!
    //! \brief In calibration Phase 1, we need to dump intermediate tensors
    //!
    bool isCalibrationDumpEnabled() const{
        return (calibrationPhase == 1);
    }

    //!
    //! \brief In calibration Phase 2, we need to build the calibration table
    //
    bool buildCalibrationCache() const{
        return (calibrationPhase == 2);
    }

    //!
    //! \brief Whether we are in calibration (either Phase 1 or 2)
    //
    bool isCalibrating() const{
        return (calibrationPhase > 0);
    }

    //!
    //! \brief Code path to build int8 graph needs to be setup when
    //! * int8Generator is being used
    //! * raw tensor data is being generated in calibration phase 1
    //! * we are building the calibration cache in phase 2
    //!
    bool useInt8ProjectionGraph() const{
        return (enableInt8Generator || isCalibrating());
    }

    //!
    //! \brief Set vocabulary file and calculate vocab size
    //!
    void setVocabulary(std::string vocabularyFile){
        vocabFile = vocabularyFile;
        vocabularySize = (getNumLinesInTextFile(vocabFile) + RESERVED_WORDS);
    }


    float getLengthPenalyMultiplier(int timeStep) const
    {
        return std::pow(5.0f + 1.0f, LENGTH_PENALTY_ALPHA) / std::pow(5.0f + timeStep, LENGTH_PENALTY_ALPHA);
    }

    void printConfig();

    //!
    //! \brief Mark a specific tensor to be printed if isValidationDumpEnabled() is true
    //! \param network The network the user is configuring
    //! \param tensor The pointer to the tensor that we would like to dump, this tensor will be replaced by this function the caller is advised to use the replaced version to make sure the plugin is not discarded by TensorRT optimizer
    //! \param name What we would like to name the tensor
    //!
    void addDebugTensor(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name, std::string unitName) const;
    
    //!
    //! \brief Mark a specific tensor to be printed if isCalibrationDumpEnabled() is true
    //! \param network The network the user is configuring
    //! \param tensor The pointer to the tensor that we would like to dump, this tensor will be replaced by this function the caller is advised to use the replaced version to make sure the plugin is not discarded by TensorRT optimizer  
    //! \param name What we would like to name the tensor
    //!
    void addCalibrationDumpTensor(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name) const;



    //!
    //! \brief Mark a specific tensor to be printed 
    //! \param network The network the user is configuring
    //! \param tensor The pointer to the tensor that we would like to dump, this tensor will be replaced by this function the caller is advised to use the replaced version to make sure the plugin is not discarded by TensorRT optimizer  
    //! \param name What we would like to name the tensor
    //!
    //! \note This adds the tensor name to mDbgTensorNames
    //! \note Make sure you call this function as earlier as possible for the tensor to be dumped and use its new value for subsequent layers (or marking it as output of the network).
    //! \note Runtime::current_batch should be >= 0 for th edump to occur as it encodes batch # into the filename.
    void addDumpPlugin(nvinfer1::INetworkDefinition* network, ITensor** tensor, std::string name) const;

public: // TBD: Need to change to private and add (public) getters and (private) setters
    int beamSize;
    std::vector<int> encoderMaxSeqLengths;
    int decoderSeqLen;
    DataType prec;
    std::string vocabFile {};
    int vocabularySize;

    //! Run the network in int8 precision
    bool enableInt8Generator;


    bool validationDump {false};
    std::vector<std::string> validateUnits {};

    int calibrationPhase {0};
    std::string calibrationData {};
    std::string calibrationCache {};

    //!
    //! Maximum batch size that can be processed.
    //! Note that this has a slightly different meaning than the concept of batchSize in Params class.
    //! The engines in GNMTCore will be built to process at most a batch size of maxBatchSize.
    //! batchSize in Params just means how we divide the input text
    //!
    int maxBatchSize {5};

    int persistentLSTMMaxBatchSize {8}; //! Maximum batch size to use persistent lstm for encoder

    // GNMT currently doesn't support different values for the following parameters
    const int RESERVED_WORDS {3};
    const int INVALID_TOKEN {-1};   // Invalid token, only used by scorer
    const int UNK_TOKEN {0};
    const int START_TOKEN {1};      // SOS token
    const int STOP_TOKEN {2};       // EOS token
    const std::string UNK {"<unk>"};
    const float LENGTH_PENALTY_ALPHA {1}; //!< The GNMT paper uses 0.6-0.7, but TF version uses 1.0.
    const int encoderLayerCount {4};
    // TBD: change name (after merge, cause this will cause conflicts in GNMTScorer)
    const int decoderLayerCount {4};
    const int hiddenSize {1024};
    const float minimalLogProb {-1.0E+37F};
};

// TBD: move these into Config
const std::string ENC_EMBED = "embeddings_encoder_embedding_encoder";
const std::string ENC_LSTM_BI_BW_BIAS = "dynamic_seq2seq_encoder_bidirectional_rnn_bw_basic_lstm_cell_bias";
const std::string ENC_LSTM_BI_BW_KERNEL = "dynamic_seq2seq_encoder_bidirectional_rnn_bw_basic_lstm_cell_kernel";
const std::string ENC_LSTM_BI_FW_BIAS = "dynamic_seq2seq_encoder_bidirectional_rnn_fw_basic_lstm_cell_bias";
const std::string ENC_LSTM_BI_FW_KERNEL = "dynamic_seq2seq_encoder_bidirectional_rnn_fw_basic_lstm_cell_kernel";
const std::string ENC_LSTM_CELL0_BIAS = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_0_basic_lstm_cell_bias";
const std::string ENC_LSTM_CELL0_KERNEL = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_0_basic_lstm_cell_kernel";
const std::string ENC_LSTM_CELL1_BIAS = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_1_basic_lstm_cell_bias";
const std::string ENC_LSTM_CELL1_KERNEL = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_1_basic_lstm_cell_kernel";
const std::string ENC_LSTM_CELL2_BIAS = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_2_basic_lstm_cell_bias";
const std::string ENC_LSTM_CELL2_KERNEL = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_2_basic_lstm_cell_kernel";
const std::string ENC_LSTM_CELL3_BIAS = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_3_basic_lstm_cell_bias";
const std::string ENC_LSTM_CELL3_KERNEL = "dynamic_seq2seq_encoder_rnn_multi_rnn_cell_cell_3_basic_lstm_cell_kernel";

const std::string ENC_LSTM_CELL4_BIAS = "ForwardPass_gnmt_encoder_with_emb_rnn_multi_rnn_cell_cell_4_lstm_cell_bias";
const std::string ENC_LSTM_CELL4_KERNEL = "ForwardPass_gnmt_encoder_with_emb_rnn_multi_rnn_cell_cell_4_lstm_cell_kernel";
const std::string ENC_LSTM_CELL5_BIAS = "ForwardPass_gnmt_encoder_with_emb_rnn_multi_rnn_cell_cell_5_lstm_cell_bias";
const std::string ENC_LSTM_CELL5_KERNEL = "ForwardPass_gnmt_encoder_with_emb_rnn_multi_rnn_cell_cell_5_lstm_cell_kernel";
const std::string ENC_LSTM_CELL6_BIAS = "ForwardPass_gnmt_encoder_with_emb_rnn_multi_rnn_cell_cell_6_lstm_cell_bias";
const std::string ENC_LSTM_CELL6_KERNEL = "ForwardPass_gnmt_encoder_with_emb_rnn_multi_rnn_cell_cell_6_lstm_cell_kernel";

const std::string ATTENTION_MEMORY_DENSE = "dynamic_seq2seq_decoder_memory_layer_kernel";
const std::string DEC_EMBED = "embeddings_decoder_embedding_decoder";
const std::string DECODE_DENSE_KERNEL = "dynamic_seq2seq_decoder_output_projection_kernel";
const std::string ATTENTION_B = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_0_attention_attention_bahdanau_attention_attention_b";
const std::string ATTENTION_G = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_0_attention_attention_bahdanau_attention_attention_g";
const std::string ATTENTION_V = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_0_attention_attention_bahdanau_attention_attention_v";
const std::string ATTENTION_QUERY_DENSE = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_0_attention_attention_bahdanau_attention_query_layer_kernel";
const std::string DEC_LSTM_CELL0_BIAS = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_0_attention_attention_basic_lstm_cell_bias";
const std::string DEC_LSTM_CELL0_KERNEL = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_0_attention_attention_basic_lstm_cell_kernel";
const std::string DEC_LSTM_CELL1_BIAS = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_1_basic_lstm_cell_bias";
const std::string DEC_LSTM_CELL1_KERNEL = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_1_basic_lstm_cell_kernel";
const std::string DEC_LSTM_CELL2_BIAS = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_2_basic_lstm_cell_bias";
const std::string DEC_LSTM_CELL2_KERNEL = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_2_basic_lstm_cell_kernel";
const std::string DEC_LSTM_CELL3_BIAS = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_3_basic_lstm_cell_bias";
const std::string DEC_LSTM_CELL3_KERNEL = "dynamic_seq2seq_decoder_multi_rnn_cell_cell_3_basic_lstm_cell_kernel";

const std::string DEC_LSTM_CELL4_BIAS = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_4_lstm_cell_bias";
const std::string DEC_LSTM_CELL4_KERNEL = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_4_lstm_cell_kernel";
const std::string DEC_LSTM_CELL5_BIAS = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_5_lstm_cell_bias";
const std::string DEC_LSTM_CELL5_KERNEL = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_5_lstm_cell_kernel";
const std::string DEC_LSTM_CELL6_BIAS = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_6_lstm_cell_bias";
const std::string DEC_LSTM_CELL6_KERNEL = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_6_lstm_cell_kernel";
const std::string DEC_LSTM_CELL7_BIAS = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_7_lstm_cell_bias";
const std::string DEC_LSTM_CELL7_KERNEL = "ForwardPass_rnn_decoder_with_attention_decoder_multi_rnn_cell_cell_7_lstm_cell_kernel";

#endif
