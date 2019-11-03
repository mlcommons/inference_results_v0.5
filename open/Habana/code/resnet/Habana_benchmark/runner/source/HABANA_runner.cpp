#include <HABANA_runner.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <iterator>
#include <loadgen.h>
#include <utility>
#include <map>
#include <cstdlib>
#include <synapse.h>
//#define CHECK_query_VALID
namespace HABANA_bench {

#define EXPECTED_NUM_OF_INPUT_DIMS      1



void Habana_singleStreamRunner::allocInputTensors()
{
      synStatus status;
      TensorMetadataInfo inputTensorInfo;
      void *synAllocPtr;
      inputTensorInfo.tensorName = m_inputName[0];
      status = synRetrieveTensorInfoByName(m_deviceId, m_topologyId, m_numOfInputs, &inputTensorInfo);
      if(status != synSuccess)
      {
           throw HABANA_benchException("Habana_singleStreamRunner, allocInputTensor(): failed to read input tensor info");
      }
      // the following check is correct only for image data
      if(inputTensorInfo.dimensions < EXPECTED_NUM_OF_INPUT_DIMS)
      {
           throw HABANA_benchException("Habana_singleStreamRunner, allocInputTensor(): Expected 4 dims [C,W,H,B]");
      }
      m_inputDims.push_back(inputTensorInfo.dimensionsSize[SYN_HEIGHT_POS]);
      m_inputDims.push_back(inputTensorInfo.dimensionsSize[SYN_WIDTH_POS]);
      m_inputDims.push_back(inputTensorInfo.dimensionsSize[SYN_DEPTH_POS]);
      m_inputSizeInBytes = m_inputDims[HEIGHT_POS] * m_inputDims[WIDTH_POS] * m_inputDims[DEPTH_POS];
      status = synMalloc(m_deviceId, m_numOfLoadedImages*m_inputSizeInBytes, synMemHost, &synAllocPtr);
      if (status != synSuccess)
      {
          synDestroy();
          throw HABANA_benchException("Habana_singleStreamRunner: can't allocate query image buffer");
      }
      m_queryBuffer = static_cast<uint8_t *>(synAllocPtr);
}


void Habana_singleStreamRunner::allocOutputTensors(bool                                         isEnforcedResBuf, 
                                                   uint64_t                                     enforcedResBufByteSize,
                                                   std::vector<std::unique_ptr<baseFunctor>>    &postProcFunctor,
                                                   uint32_t                                     numOfTasks)
{
      std::vector<Habana_taskInfo>::iterator taskInfoIt;
      bool outputAllocStatus=true;
 
      synStatus status;
      
      if((isEnforcedResBuf) && (enforcedResBufByteSize == 0))
      {
           synFree(m_deviceId, m_queryBuffer, synMemHost); // the queryBuffer was allocated during allocInputTensor()
           synDestroy(); 
           throw HABANA_benchException("Habana_singleStreamRunner, allocOutputTensor(): failed to read output tensor info");
      }
      for(int i = 0; i < m_numOfOutputs;i++)
      {
        uint64_t tensorByteSize;
        m_outputTensorInfo[i].tensorName = m_outputName[i];
      }
      status = synRetrieveTensorInfoByName(m_deviceId, m_topologyId, m_numOfOutputs, m_outputTensorInfo);
      if(status != synSuccess)
      {
           synFree(m_deviceId, m_queryBuffer, synMemHost); // the queryBuffer was allocated during allocInputTensor()
           synDestroy(); 
           throw HABANA_benchException("Habana_singleStreamRunner, allocOutputTensor(): failed to read output tensor info");
      }
      try
      {
           for(int32_t i = 0; i < numOfTasks;i++)
           {
               m_tasks.push_back(Habana_taskInfo(isEnforcedResBuf,
                                    enforcedResBufByteSize,
                                    m_deviceId,
                                    m_batchSize,
                                    m_numOfOutputs,
                                    m_outputTensorInfo,
                                    std::move(postProcFunctor[i]),
                                    m_inputName[0],
                                    m_inputSizeInBytes));
           }
      }
      catch(const HABANA_benchException& e)
      {
          synFree(m_deviceId, m_queryBuffer, synMemHost); // the queryBuffer was allocated during allocInputTensor()
          m_tasks.clear();
          synDestroy();
          throw;
      }
      catch(const std::exception& e)
      {
          synFree(m_deviceId, m_queryBuffer, synMemHost); // the queryBuffer was allocated during allocInputTensor()
          m_tasks.clear();
          synDestroy();
          throw;
      }
      
}


Habana_singleStreamRunner::Habana_singleStreamRunner(const std::string                          &recipeName,
                                                     const std::string                          &imageCacheDirName,
                                                     const std::string                          &imageNameList,
                                                     uint64_t                                   maxNumOfquerySamples,
                                                     uint64_t                                   numberOfImageToLoad,
                                                     uint64_t                                   batchSize,
                                                     std::vector<std::unique_ptr<baseFunctor>>  &outputTransProc,
                                                     bool                                       isEnforcedResBuff,
                                                     uint64_t                                   enforcedResBufByteSize,
                                                     bool                                       enableOutputDump,
                                                     uint32_t                                   outputDumpHaltCnt,
                                                     uint32_t                                   numOfTasks) : 
                                                                                                m_maxNumOfquerySample(maxNumOfquerySamples),
                                                                                                m_numOfLoadedImages(numberOfImageToLoad),
                                                                                                m_batchSize(batchSize),
                                                                                                m_numOfTasks(numOfTasks),
                                                                                                m_enableOutputDump(enableOutputDump),
                                                                                                m_outputDumpHaltCnt(outputDumpHaltCnt)
{
      synStatus status;
      void *synAllocPtr;
      bool outputAllocStatus = true;
      char tempName[1][ENQUEUE_TENSOR_NAME_MAX_SIZE];
      status = synInitialize(); // init the synapse API
      if (status != synSuccess) 
      {
        throw HABANA_benchException("Habana_singleStreamRunner: Failed to init synapse");
      }
      status = synAcquireDevice(&m_deviceId, nullptr); // aquire a habana HW device
      if (status != synSuccess)
      {
        synDestroy();
        throw HABANA_benchException("Habana_singleStreamRunner: Failed to get device");
      }
      // load a habana goya recipe file (model file)
      status = synLoadRecipe(m_deviceId, recipeName.c_str(), &m_topologyId);
      if (status != synSuccess)
      {
        std::string error("Habana_singleStreamRunner failed to load recipe file: ");
        error = error + recipeName;
        throw HABANA_benchException(error);
      }
      // get the active topologe ID
      status = synActivateTopology(m_deviceId, m_topologyId);
      if (status != synSuccess)
      {
        throw HABANA_benchException("Habana_singleStreamRunner to load topolegy");
      }

      status = synGetIOTensorsAmount(m_deviceId, m_topologyId, m_numOfInputs,m_numOfOutputs, m_numOfInter);
      if(status != synSuccess)
      {
          synDestroy();
          throw HABANA_benchException("Habana_singleStreamRunner: can get num of input/output from the device");
      }
      if((m_numOfInputs != 1) || (m_numOfOutputs > 4))
      {
          synDestroy();
          throw HABANA_benchException("Habana_singleStreamRunner: number of reported input/output exceeds the allowed number");
      }

      status =  synGetTensorsName(m_deviceId, m_topologyId, m_inputName, m_numOfInputs, m_outputName, m_numOfOutputs, tempName, 0);
      if(status != synSuccess)
      {
          synDestroy();
          throw HABANA_benchException("Habana_singleStreamRunner: Couldn't read input/output tensors names");
      }
     
      
      

      allocInputTensors(); // may throw exception upon failure to read data from device or allocation problem

      try
      {
            // init the image cache manager
          m_runnerImageCacheManager.reset(new HABANA_imageCacheManager(imageCacheDirName,
                                                                       imageNameList,
                                                                       maxNumOfquerySamples,
                                                                       m_numOfLoadedImages,
                                                                       m_inputDims[HEIGHT_POS],
                                                                       m_inputDims[WIDTH_POS],
                                                                       m_inputDims[DEPTH_POS],
                                                                       m_queryBuffer));
      }
      catch (const HABANA_benchException &e)
      {
          synFree(m_deviceId, m_queryBuffer, synMemHost);
          synDestroy();
          throw;
      }
      allocOutputTensors(isEnforcedResBuff,enforcedResBufByteSize,outputTransProc,m_numOfTasks);
}

Habana_singleStreamRunner::~Habana_singleStreamRunner() 
{
    m_tasks.clear();
    synFree(m_deviceId, m_queryBuffer, synMemHost);
    synDestroy();
}



void Habana_singleStreamRunner::loadSamplesToRam(uintptr_t                      clientData,
                                                 const mlperf::QuerySampleIndex *querySamplesIndex,
                                                  size_t numOfquerySamples)
{
    m_runnerImageCacheManager->load_samples(querySamplesIndex, numOfquerySamples);
}

void Habana_singleStreamRunner::unloadSamplesFromRam(uintptr_t clientData,
                                                     const mlperf::QuerySampleIndex *querySamplesIndex,
                                                     size_t numOfquerySamples)
{
    m_runnerImageCacheManager->unload_samples(numOfquerySamples);
}


void  Habana_singleStreamRunner::dumpRawOutputs(Habana_taskInfo &taskInfo, uint64_t halt_cnt)
{
    static uint64_t cnt = 0;
    std::ofstream imageNames;
    std::ofstream outputs;
    std::vector<mlperf::QuerySample>::iterator querySampIt;
    std::string imageFileName;
    if(halt_cnt < cnt)
        return;
    if(cnt == 0)
    {
        const int dir_err = system("mkdir -p dbg");
        if (-1 == dir_err)
        {
            throw HABANA_benchException("Habana_singelThreadRunner: Error creating output denug dump directory");
        }
    }
    for(querySampIt = taskInfo.m_querySamples.begin(); querySampIt != taskInfo.m_querySamples.end();querySampIt++)
    {
        std::string fileName = m_runnerImageCacheManager->getImageName(querySampIt->index);
        imageFileName = "dbg/image_name_dump_" + std::to_string(querySampIt->index) + ".txt";
        imageNames.open(imageFileName.c_str());
        imageNames<<fileName<<std::endl;
        imageNames.close();
        for(int i=0;i < m_numOfOutputs;i++)
        {
            std::string outputName(taskInfo.m_enqueueOutputTensors[i].tensorName);
            outputName = "dbg/" + outputName + "_" + std::to_string(querySampIt->index) +".bin";
            outputs.open(outputName.c_str(),std::ios::binary);
            outputs.write(taskInfo.m_enqueueOutputTensors[i].pTensorData,taskInfo.m_enqueueOutputTensors[i].tensorSize);
            outputs.close();
        }
    }
    cnt++;
}



void Habana_singleStreamRunner::queueQuery(uintptr_t                    ClientData,
                                           const mlperf::QuerySample    *querySamples,
                                           size_t                       numOfquerySamples)
{
    uint8_t * inDataPtr;
    uint32_t inDataSize;
    synWaitHandle handle;
    synStatus status;
    for(int i=0; i < numOfquerySamples;i++)
    {
        m_tasks[0].m_querySamples.push_back(*querySamples); // push back the new sample into the query smaple vector
        querySamples++;
    }
                                                        // of the task
    m_runnerImageCacheManager->getNextBatch(m_tasks[0]); // get sample memory pointer (updated into the task
                                                        // structure)
    // enqueue the sample to be processed on the Habana HW
    status = synEnqueueByName(m_deviceId,
                              &m_tasks[0].m_enqueueInputTensor,
                              m_numOfInputs,
                              m_tasks[0].m_enqueueOutputTensors,
                              m_numOfOutputs,
                              &handle);                      
    if (status != synSuccess)
    {
        throw HABANA_benchException("Habana_singelThreadRunner: Error in Synapse enqueue function");
    }
    else 
    {
      synWaitForEvent(m_deviceId, handle); // wait for the HW to finish processing the sample
      if(m_enableOutputDump)
            dumpRawOutputs(m_tasks[0],m_outputDumpHaltCnt);
      sutReportResults(m_tasks[0]); // translate and report results to loadgen
    }
}

HABANA_benchStatus Habana_singleStreamRunner::sutReportResults(Habana_taskInfo &task) 
{
    std::vector<uint32_t>::iterator partIt;
    // cast the pointer into uint64 (to be passed to loadgen)
    uintptr_t base_ptr = reinterpret_cast<uintptr_t>(task.m_floatTranslationbuffer[0]);
    mlperf::QuerySampleResponse newResponse;
    // call the translation callback which return a vector of number of results
    // element for each sample
    if (task.m_postProcFactorPtr->operator()(task.m_resPartitions, task.m_querySamples.size(), task.m_enqueueOutputTensors, task.m_floatTranslationbuffer,task.m_querySamples) == false)
          return HABANA_benchStatus::HABANA_FAIL;
    for (int i = 0; i < task.m_querySamples.size(); i++) 
    {
      // generate QuerySampleResponse structure and push it into the query
      // response vector
      task.m_querySamplesResponseBuff.push_back({task.m_querySamples[i].id, base_ptr,task.m_resPartitions[i] * sizeof(float)});
      // advnce the pointer to the next sample
      base_ptr += (task.m_resPartitions[i] * sizeof(float));
    }
    // pass results to loadgen - this function  informs the loadgen to set the
    // time count for this query
    mlperf::QuerySamplesComplete(task.m_querySamplesResponseBuff.data(),task.m_querySamples.size());
    // clear all vectors for next task reuse
    task.m_querySamplesResponseBuff.clear();
    task.m_querySamples.clear();
    task.m_resPartitions.clear();
    return HABANA_benchStatus::HABANA_SUCCESS;
}

void Habana_singleStreamRunner::flushQueries() {}
void Habana_singleStreamRunner::reportLatencyReport(uintptr_t ClientData,
                                                    const int64_t *timeBuffer,
                                                    size_t len) {}

Habana_multiStreamRunner::Habana_multiStreamRunner(const std::string                            &recipeName,
                                                   const std::string                            &imageCacheDirName,
                                                   const std::string                            &imageNameList,
                                                   uint64_t                                     maxNumOfquerySamples,
                                                   uint64_t                                     numberOfImageToLoad,
                                                   uint64_t                                     batchSize,
                                                   std::vector<std::unique_ptr<baseFunctor>>    &outputTransProc,
                                                   bool                                         isEnforcedResBuff,
                                                   uint64_t                                     enforcedResBufByteSize,
                                                   bool                                         enableOutputDump,
                                                   uint32_t                                     outputDumpHaltCnt,
                                                   uint32_t                                     numOfTasks) :
                                Habana_singleStreamRunner(recipeName,
                                                          imageCacheDirName,
                                                          imageNameList,
                                                          maxNumOfquerySamples,
                                                          numberOfImageToLoad,
                                                          batchSize,
                                                          outputTransProc,
                                                          isEnforcedResBuff,
                                                          enforcedResBufByteSize,
                                                          enableOutputDump,
                                                          outputDumpHaltCnt,
                                                          numOfTasks)
{
  try
  {
      m_enqueueThreads.resize(numOfTasks);
  } 
  catch (const std::exception &e) 
  {
    throw;
  }
}

Habana_multiStreamRunner::~Habana_multiStreamRunner() 
{
}
void    Habana_multiStreamRunner::queueSingleBatchQuery(uintptr_t ClientData, const mlperf::QuerySample *querySamples, size_t numOfquerySamples)
{
      int i = 0;
      synStatus status;
      while (m_enqueueThreads[i].isBusy()) 
      {
        ++i;
        i = i % m_numOfTasks;
      }
      Habana_taskInfo &taskInfo = m_tasks[i];
      for (int j = 0; j < numOfquerySamples; j++) 
      {
          taskInfo.m_querySamples.push_back(*querySamples);
          querySamples++;
      }
      m_runnerImageCacheManager->getNextBatch(taskInfo);

      status = synEnqueueByName(m_deviceId,
                                &taskInfo.m_enqueueInputTensor,
                                m_numOfInputs,
                                taskInfo.m_enqueueOutputTensors,
                                m_numOfOutputs,
                                &taskInfo.m_enqueueHandle);         
      if (status != synSuccess)
      {
            throw HABANA_benchException("Habana_multiThreadRunner: Error in Synapse enqueue function");
      } 
      m_enqueueThreads[i].sendTaskCpy([this,&taskInfo]() {process(taskInfo);},[](){});

}

void Habana_multiStreamRunner::queueQuery(uintptr_t ClientData, const mlperf::QuerySample *querySamples, size_t numOfquerySamples) 
{
      queueSingleBatchQuery(ClientData, querySamples, numOfquerySamples);
}

void Habana_multiStreamRunner::process( Habana_taskInfo &taskInfo)
{
      synWaitForEvent(m_deviceId, taskInfo.m_enqueueHandle);
      sutReportResults(taskInfo);
}









Habana_offlineRunner::Habana_offlineRunner(const std::string                          &recipeName,
                                           const std::string                          &imageCacheDirName,
                                           const std::string                          &imageNameList,
                                           uint64_t                                   maxNumOfquerySamples,
                                           uint64_t                                   numberOfImageToLoad,
                                           uint64_t                                   batchSize,
                                           std::vector<std::unique_ptr<baseFunctor>>  &outputTransProc, 
                                           bool                                       isEnforcedResBuff,
                                           uint64_t                                   enforcedResBufByteSize,
                                           bool                                       enableOutputDump,
                                           uint32_t                                   outputDumpHaltCnt,
                                           uint32_t                                   numOfTasks,
                                           uint32_t                                   breakedBatchSize) : Habana_multiStreamRunner(recipeName,
                                                                                                                                  imageCacheDirName,
                                                                                                                                  imageNameList,
                                                                                                                                  maxNumOfquerySamples,
                                                                                                                                  numberOfImageToLoad,
                                                                                                                                  batchSize,
                                                                                                                                  outputTransProc,
                                                                                                                                  isEnforcedResBuff,
                                                                                                                                  enforcedResBufByteSize,
                                                                                                                                  enableOutputDump,
                                                                                                                                  outputDumpHaltCnt,
                                                                                                                                  numOfTasks)
{
    m_breakBatchSize = breakedBatchSize;
}
Habana_offlineRunner::~Habana_offlineRunner(){}

void    Habana_offlineRunner::queueQuery(uintptr_t ClientData, const mlperf::QuerySample* queryPtr, size_t queryLen)
{
      size_t i=0, batchSize;
      while(i < queryLen)
      {
          if((queryLen - i) > m_breakBatchSize)
              batchSize = m_breakBatchSize;
          else
              batchSize = queryLen - i;
          i+= batchSize;
          queueSingleBatchQuery(ClientData, queryPtr, batchSize);
          queryPtr += m_breakBatchSize;
      }
}    

Habana_serverRunner::Habana_serverRunner(const std::string                          &recipeName,
                                         const std::string                          &imageCacheDirName,
                                         const std::string                          &imageNameList,
                                         uint64_t                                   maxNumOfquerySamples,
                                         uint64_t                                   numberOfImageToLoad,
                                         uint64_t                                   batchSize,
                                         std::vector<std::unique_ptr<baseFunctor>>  &outputTransProc, 
                                         bool                                       isEnforcedResBuff,
                                         uint64_t                                   enforcedResBufByteSize,
                                         bool                                       enableOutputDump,
                                         uint32_t                                   outputDumpHaltCnt,
                                         uint32_t                                   numOfTasks,
                                         std::chrono::duration<int,std::micro>      expectedImageProc,
                                         std::chrono::duration<int,std::micro>      procSlotDurationTime) : Habana_multiStreamRunner(recipeName,
                                                                                                                                  imageCacheDirName,
                                                                                                                                  imageNameList,
                                                                                                                                  maxNumOfquerySamples,
                                                                                                                                  numberOfImageToLoad,
                                                                                                                                  batchSize,
                                                                                                                                  outputTransProc,
                                                                                                                                  isEnforcedResBuff,
                                                                                                                                  enforcedResBufByteSize,
                                                                                                                                  enableOutputDump,
                                                                                                                                  outputDumpHaltCnt,
                                                                                                                                  numOfTasks)
{
    void *synAllocPtr;
    synStatus status;
    m_batchifierDataVect.reserve(numOfTasks);

    for(uint32_t i=0; i <numOfTasks;i++)
    {
        status = synMalloc(m_deviceId, batchSize*m_inputSizeInBytes, synMemHost, &synAllocPtr); 
        m_batchifierDataVect.push_back(Hababan_batchifierData(reinterpret_cast<char *>(synAllocPtr),m_inputSizeInBytes,m_batchSize,expectedImageProc,procSlotDurationTime));       
    }
    m_procSlotDurationTime = procSlotDurationTime;
    m_watchDogEnable    = true;
    m_watchDogThread = std::thread(&Habana_serverRunner::watchDog,this);
}
Habana_serverRunner::~Habana_serverRunner()
{
    m_watchDogEnable = false;
    while(!m_watchDogThread.joinable())
    {

    }
    m_watchDogThread.join();
}
void    Habana_serverRunner::queueQuery(uintptr_t ClientData, const mlperf::QuerySample* querySamples, size_t len)
{
    int i=0;
    {
        
        while (m_enqueueThreads[i].isBusy())
        {
            ++i;
            i = i % m_numOfTasks;
        }
    }
    
    Habana_taskInfo         &taskInfo = m_tasks[i];
    Hababan_batchifierData  &batchfierData = m_batchifierDataVect[i];
   
    taskInfo.m_querySamples.push_back(*querySamples);
    batchfierData.lock();
    m_enqueueThreads[i].sendTaskCpy([this,i,querySamples]() {batchfier(i,querySamples);},[](){});
    batchfierData.unlock();
}


void Habana_serverRunner::enqueueAndProc(int i)
{
        synStatus status;
        Habana_taskInfo &taskInfo = m_tasks[i];
        Hababan_batchifierData &batchifierData = m_batchifierDataVect[i];
        taskInfo.updateSizeIO(batchifierData.getNumOfImages(),batchifierData.get_ptr());
        status = synEnqueueByName(m_deviceId,
                                  &taskInfo.m_enqueueInputTensor,
                                  m_numOfInputs,
                                  taskInfo.m_enqueueOutputTensors,
                                  m_numOfOutputs,
                                  &taskInfo.m_enqueueHandle);         
        if (status != synSuccess)
        {
            throw HABANA_benchException("Habana_serverRunner: Error in Synapse enqueue function");
        } 
        batchifierData.clear();
        process(taskInfo);
}


void Habana_serverRunner::batchfier(int i,const mlperf::QuerySample* querySamples)
{
    
    Hababan_batchifierData &batchifierData = m_batchifierDataVect[i];
    batchifierData.lock();
    if(batchifierData.copyImage(m_runnerImageCacheManager->getSample(querySamples)))
    {
       enqueueAndProc(i);
    }
    batchifierData.unlock();
}
void Habana_serverRunner::watchDog()
{
    while (m_watchDogEnable)
    {
        std::this_thread::sleep_for(m_procSlotDurationTime);
        for(int i = 0; i <m_numOfTasks;i++)
        {
            
            if(m_batchifierDataVect[i].isBatchOverdue())
            {
                m_batchifierDataVect[i].lock();
                if(m_batchifierDataVect[i].isBatchOverdue())
                    enqueueAndProc(i);
                m_batchifierDataVect[i].unlock();
            }
        }
    }
}
} // namespace HABANA_bench

