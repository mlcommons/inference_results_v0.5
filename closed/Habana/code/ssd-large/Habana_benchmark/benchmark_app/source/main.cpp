#include <HABANA_runner_c_interface.h>
#include <c_api.h>
#include <iostream>
#include <fstream>
#include <test_settings.h>
#include <math.h>
#include <INIReader.h>
#include <vector>
#include <omp.h>
#include <query_sample.h>
#include <HABANA_transProcessFunctors.h>
#include <boxes.h>

#define MS_IN_SECOND 1000

using namespace HABANA_bench;

enum class ssdThreadingType
{
    OPENMP,
    TASK
};


#define MAX_NUM_OF_BENCHMARK_THREADS        4
#define MAX_NUM_OF_SSD_THREADS              1
#define DEFUALT_DUMP_HALT_CNT               10
#define NUM_OF_SINGLESTREAM_THREADS         1
#define NUM_OF_OFFLINE_THREADS              1
#define SCALE_XY                        0.1f
#define SCALE_WH                        0.2f
#define CRITERIA                        0.5f
#define THRESHOLD			            (1638) //32768*0.05

enum class modelType
{
    RESNET50,
    SSD
};








//************************************************************************************************************
//************************************************************************************************************
// readConfiguration - the function reads an ini file and fill loadgen test setting structure.
// the function relies on INIReader open source which generates a map of [key,value]
// input:
// initFileName - ini input file name
// habanaTestSettings - loadgen test setting structure to be init
// 
// output:
// PASS/FAIL according to the ini file parsing
//************************************************************************************************************
//************************************************************************************************************

HABANA_benchStatus readConfiguration(std::string          initFileName,
                                     mlperf::TestSettings &testSettings,
                                     std::string          &recipeFileName,
                                     std::string          &imageDirName,
                                     std::string          &listFileName,
                                     modelType            &runModelType,
                                     bool                 &enableOutputDump,
                                     size_t               &haltDump,
                                     size_t               &numOfThreads,
                                     size_t               &numOfSSDThreads,
                                     ssdThreadingType     &threadingType,
                                     uint32_t             &breakBatchSize,
                                     uint64_t             &numberOFImagesToLoad,
                                     uint64_t             &numberOfImagesForPerformance,
                                     int32_t              &expectedImageProc,
                                     int32_t              &serverDelay)
{
    INIReader habanaIni(initFileName.c_str());
    serverDelay = 0;
    expectedImageProc = 0;
    std::string baseMlperfConfigPath;
    std::string userMlperfConfigPath;
    std::string auditMlperfConfigPath;
    std::string modelTypeName;
    std::string scenario;
    
    int cfgRetVal = 0;
    if(habanaIni.ParseError() == -1)
    {
        std::cout<<"Ini file parsing error, please check file name or file content";
        exit(1);
    }
    std::string   stringVal;
    imageDirName = habanaIni.GetString("general","image_dir","");
    if(imageDirName =="")
    {
        std::cout<<"image path is missing"<<std::endl;
        exit(1);
    }
    listFileName = habanaIni.GetString("general","list_file_name","");
    if(listFileName == "")
    {
        std::cout<<"image list file is missing"<<std::endl;
        exit(1);
    }
    scenario = habanaIni.GetString("general","scenario","");
    if(scenario == "SingleStream")
    {
        testSettings.scenario = mlperf::TestScenario::SingleStream;
    } 
    else if(scenario == "MultiStream")
    {
        testSettings.scenario = mlperf::TestScenario::MultiStream;
    }
    else if(scenario == "Server")
    {
        testSettings.scenario = mlperf::TestScenario::Server;
    } 
    else if(scenario == "Offline")
    {
        testSettings.scenario = mlperf::TestScenario::Offline;
    }
    else
    {
            std::cout<<"Undefined run scenario: "<<stringVal<<::std::endl;
            std::cout<<"should be one of the following: SingleStream, MultiStream, Server, Offline"<<std::endl;
            exit(1);
    }   
    
    
    baseMlperfConfigPath = habanaIni.GetString("general","base_mlperf_config_path","");
    if(baseMlperfConfigPath == "")
    {
        std::cout<<"base mlperf config file path is missing"<<std::endl;
        exit(1);
    }
    userMlperfConfigPath = habanaIni.GetString("general","user_mlperf_config_path","");
    if(userMlperfConfigPath == "")
    {
        std::cout<<"user mlperf config file path is missing"<<std::endl;
        exit(1);
    }

    auditMlperfConfigPath = habanaIni.GetString("general","audit_mlperf_config_path","");
    if(auditMlperfConfigPath == "")
    {
        std::cout<<"Audit mlperf config file path is missing"<<std::endl;
    }
    modelTypeName = habanaIni.GetString("general","model_type","");
    if(modelTypeName == "resnet50")
        runModelType = modelType::RESNET50;
    else if(modelTypeName =="ssd-resnet34")
    {
        runModelType = modelType::SSD;
    }
    else
    {
        std::cout<<"undefined model type: "<<modelTypeName<<std::endl;
        std::cout<<"should be on of the following: resnet50 or ssd-resnet34"<<std::endl;
        exit(2);
    }
    
    recipeFileName = habanaIni.GetString("general","recipe_file_name","");
    if(recipeFileName == "")
    {
        std::cout<<"Recipe file name is missing"<<std::endl;
        exit(1);
    }
    cfgRetVal = testSettings.FromConfig(baseMlperfConfigPath,modelTypeName,scenario);
    if(cfgRetVal != 0)
    {
        std::cout<<"couldn't config test setting struct with base configuration file, FromConfig ret val: "<<cfgRetVal<<std::endl;
        exit(1);
    }
    cfgRetVal = testSettings.FromConfig(userMlperfConfigPath,modelTypeName,scenario);
    if(cfgRetVal != 0)
    {
        std::cout<<"couldn't config test setting struct with user configuration file, FromConfig ret val: "<<cfgRetVal<<std::endl;
        exit(1);
    }
    if(auditMlperfConfigPath != "")
    {
        cfgRetVal = testSettings.FromConfig(auditMlperfConfigPath,modelTypeName,scenario);
        if(cfgRetVal != 0)
        {
            std::cout<<"couldn't config test setting struct with audit configuration file, FromConfig ret val: "<<cfgRetVal<<std::endl;
        }
    }
    if(scenario == "Server")
    {
        serverDelay =  testSettings.server_target_latency_ns/MS_IN_SECOND;
        if(serverDelay <= 0)
        {
            std::cout<<"expected server delay must be larger than 0"<<std::endl;
        }
    }

    enableOutputDump = habanaIni.GetBoolean("general","output_tensor_dump_enable",false);
    haltDump = habanaIni.GetInteger("general","output_tensor_dump_halt_cnt",DEFUALT_DUMP_HALT_CNT);
    numOfThreads = habanaIni.GetInteger("general","num_of_threads",MAX_NUM_OF_BENCHMARK_THREADS);
    numOfSSDThreads = habanaIni.GetInteger("general","num_of_ssd_threads",MAX_NUM_OF_SSD_THREADS);
    breakBatchSize = habanaIni.GetInteger("general","break_batch_size",200);
    numberOFImagesToLoad  = habanaIni.GetInteger("general","num_of_images_to_load",4952);
    numberOfImagesForPerformance = habanaIni.GetInteger("general","num_of_images_for_performance",1024);
    expectedImageProc= habanaIni.GetInteger("general","expected_processing_latency",0);
    if(scenario == "Server")
    {
        if(expectedImageProc <= 0)
        {
                std::cout<<"expected sample process time must be larger than 0"<<std::endl;
        }
    }
    stringVal = habanaIni.GetString("general","ssd_threading_type","TSK");
    
    if(stringVal == "OMP")
        threadingType = ssdThreadingType::OPENMP;
    else if(stringVal == "TSK")
        threadingType = ssdThreadingType::TASK;
    else
    {
        std::cout<<"unknow ssd threading type value"<<std::endl;
        exit(1);
    }
   
    return HABANA_benchStatus::HABANA_SUCCESS; 
}



void initModelData(std::vector<std::unique_ptr<baseFunctor>>  &postProcFunactors,
                   size_t                                     numOfBenchmarkThreads,
                   size_t                                     numOfSSDThreads,
                   size_t                                     batchSize,
                   modelType                                  modelTypeIn,
                   bool                                       &isEnforcedTansBuffer,
                   size_t                                     &enforcedBufferSize,
                   ssdThreadingType                           ssdThreadingTypeIn)
{
    switch (modelTypeIn)
    {
        case modelType::RESNET50:
            isEnforcedTansBuffer    = false;
            enforcedBufferSize      = 0;
            postProcFunactors.reserve(numOfBenchmarkThreads); 
            for(int i = 0; i < numOfBenchmarkThreads;i++)  
                postProcFunactors.push_back(std::unique_ptr<baseFunctor>(new resnetTransOperator()));      
            break;
        case modelType::SSD:

            isEnforcedTansBuffer    = true;
            enforcedBufferSize      = batchSize * MAX_NUM*NUM_OF_ELEMENTS_IN_SSD_RESULT*sizeof(float);
            postProcFunactors.reserve(numOfBenchmarkThreads); 
            if(ssdThreadingTypeIn == ssdThreadingType::OPENMP)
            {
                std::cout<<"using OPENNMP"<<std::endl;
                omp_set_num_threads(numOfSSDThreads);
                size_t thnum = omp_get_max_threads();
                std::cout<<"Num of openmp threads: "<<thnum<<std::endl;
                for(int i = 0; i < numOfBenchmarkThreads;i++)
                    postProcFunactors.push_back(std::unique_ptr<baseFunctor>(new ssdTransOperatorSSDOmp(MAX_NUM,thnum,base_boxes,SCALE_XY,SCALE_WH,CRITERIA,THRESHOLD)));
            }
            else if(ssdThreadingTypeIn == ssdThreadingType::TASK)
            {
                std::cout<<"using THREADED_TASK"<<std::endl;
                for(int i = 0; i < numOfBenchmarkThreads;i++)
                    postProcFunactors.push_back(std::unique_ptr<baseFunctor>(new ssdTransOperatorSSDTsk(MAX_NUM,numOfSSDThreads,base_boxes,SCALE_XY,SCALE_WH,CRITERIA,THRESHOLD)));
            }
            else
            {
                std::cout<<"initModelData: unknown SSD threading type"<<std::endl;
                exit(1);
            }
            break;
        
        default:
            std::cout<<"Unknown model supplied";
            exit(1);
            break;
    }
}

size_t initScenario(mlperf::TestSettings &testsettings, size_t &numOfBenchmarkThreads, uint32_t breakBatchSize, int32_t expectedImageProcTime, modelType runModelType)
{
    switch(testsettings.scenario)
    {
        case mlperf::TestScenario::SingleStream:
            numOfBenchmarkThreads = NUM_OF_SINGLESTREAM_THREADS;
            return 1;
        case mlperf::TestScenario::MultiStream:
            return  testsettings.multi_stream_samples_per_query;
        case mlperf::TestScenario::Server:
            if(runModelType == modelType::RESNET50)
                return ((testsettings.server_target_latency_ns/MS_IN_SECOND/expectedImageProcTime));
            else
                return 1;            
        case mlperf::TestScenario::Offline:
            return breakBatchSize;
        case mlperf::TestScenario::MultiStreamFree:
            return testsettings.multi_stream_samples_per_query;
        default:
            std::cout<<"initScenario: Unsupported TestScenario";
            exit(1);
    }
}

int main(int argc, char *argv[])
{
    std::string         recipeName;
    std::string         imageDirePath;
    std::string         imagListFile;
    size_t              batchSize = 1;
    void                *sutPtr;
    void                *qslPtr;
    modelType           runModelType;
    bool                isEnforcedResBuf=false;
    size_t              enforcedBufSize = 0;
    bool                enableOutputDump=false;
    size_t              haltCnt = DEFUALT_DUMP_HALT_CNT;
    size_t              numOfBenchmarkThreads;
    size_t              numOfSSDThreads;
    ssdThreadingType    ssdThrType;
    uint32_t            breakBatchSize;
    uint64_t            numberOfImagesToLoad;
    uint64_t            numberOfImagesForPerformance;
    int32_t            expectedImageProcTime;
    int32_t            serverDelay;
    mlperf::TestSettings HABANA_testsettings;
    mlperf::LogSettings  HABANA_logSettings;
    HABANA_logSettings.enable_trace = false;
    std::vector<std::unique_ptr<baseFunctor>> postProcFunctors;
    if(argc < 2)
    {
        std::cout<<"INI files is needed."<<std::endl;
        std::cout<<"usage: "<<std::endl;
        std::cout<<"HABANA_benchmark_app ini_file_name"<<std::endl;
        exit(1);
    }
    readConfiguration(argv[1],
                      HABANA_testsettings,
                      recipeName,
                      imageDirePath,
                      imagListFile,
                      runModelType,
                      enableOutputDump,
                      haltCnt,
                      numOfBenchmarkThreads,
                      numOfSSDThreads,
                      ssdThrType, 
                      breakBatchSize,
                      numberOfImagesToLoad,
                      numberOfImagesForPerformance,
                      expectedImageProcTime,
                      serverDelay);
    batchSize =  initScenario(HABANA_testsettings, numOfBenchmarkThreads,breakBatchSize,expectedImageProcTime,runModelType);
    initModelData(postProcFunctors, numOfBenchmarkThreads, numOfSSDThreads, batchSize, runModelType, isEnforcedResBuf, enforcedBufSize,ssdThrType);
    

    
    
    std::cout<<"Init Runner"<<std::endl;
    try
    {
            if(runModelType == modelType::RESNET50)
            {
                runnerAllocRunner(HABANA_testsettings.scenario,
                                recipeName,
                                imageDirePath,
                                imagListFile,
                                HABANA_testsettings.min_query_count,
                                numberOfImagesToLoad + SAFE_GUARD_LEN_RESNET50,
                                batchSize,
                                postProcFunctors,
                                enableOutputDump,
                                haltCnt,
                                numOfBenchmarkThreads,
                                breakBatchSize,
                                std::chrono::microseconds(expectedImageProcTime),
                                std::chrono::microseconds(serverDelay),
                                isEnforcedResBuf,
                                enforcedBufSize);
            }
            else
            {
                runnerAllocRunner(HABANA_testsettings.scenario,
                                recipeName,
                                imageDirePath,
                                imagListFile,
                                HABANA_testsettings.min_query_count,
                                numberOfImagesToLoad + SAFE_GUARD_LEN_SSD,
                                batchSize,
                                postProcFunctors,
                                enableOutputDump,
                                haltCnt,
                                numOfBenchmarkThreads,
                                breakBatchSize,
                                std::chrono::microseconds(expectedImageProcTime),
                                std::chrono::microseconds(serverDelay),
                                isEnforcedResBuf,
                                enforcedBufSize);  
            }
            

    }

    catch(HABANA_benchException &e)
    {
        std::cout<<e.what()<<std::endl;
        exit(1);
    }
    std::cout<<"allocate SUT and QSL"<<std::endl;
    sutPtr = mlperf::c::ConstructSUT(0,"HABANA_Runner",sizeof("HABANA_Runner"),runnerQueueQuery,runnerFlushQuries,runnerReportLatencyReport);
    qslPtr = mlperf::c::ConstructQSL(0,"HABANA_Runner",sizeof("HABANA_Runner"),numberOfImagesToLoad,numberOfImagesForPerformance,runnerLoadSamplesToRam,runnerUnloadSamplesFromRam);
    std::cout<<"Start test"<<std::endl;

    mlperf::c::StartTest(sutPtr,qslPtr,HABANA_testsettings,HABANA_logSettings);
    mlperf::c::DestroyQSL(qslPtr);
    mlperf::c::DestroySUT(sutPtr);
    std::cout<<"Before destroySynapse"<<std::endl;
    freeRunnerInterface();
    std::cout<<"after destroySynapse"<<std::endl;
    return 0;
}