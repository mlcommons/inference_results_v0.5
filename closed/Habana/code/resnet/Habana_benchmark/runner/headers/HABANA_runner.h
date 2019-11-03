#ifndef HABANA_RUNNER
#define HABANA_RUNNER
#include <query_sample.h>
#include <query_sample_library.h>
#include <vector>
#include <stdint.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <HABANA_exception.h>
#include <HABANA_transProcessBaseFunctors.h>
#include <HABANA_runner_task.h>
#include <HABANA_imageCacheManager.h>
#include <HABANA_threadedTask.h>
#include <chrono>
#include <HABANA_batchifierDataNode.h>
namespace HABANA_bench
{

    

    enum dimPositions
    {
        HEIGHT_POS,
        WIDTH_POS,
        DEPTH_POS
    };    

    enum synapseTensorDimPosition
    {
        SYN_DEPTH_POS,
        SYN_WIDTH_POS,
        SYN_HEIGHT_POS,
        SYN_BATCH_POS
    };
        
    class Habana_singleStreamRunner
    {
        public:
                                
                                Habana_singleStreamRunner(const std::string                             &recipeName,
                                                          const std::string                             &imageCacheDirName,
                                                          const std::string                             &imageNameList,
                                                          uint64_t                                      maxNumOfquerySamples,
                                                          uint64_t                                      numberOfImageToLoad,
                                                          uint64_t                                      batchSize,
                                                          std::vector<std::unique_ptr<baseFunctor>>     &outputTransProc, 
                                                          bool                                          isEnforcedResBuff,
                                                          uint64_t                                      enforcedResBufByteSize,
                                                          bool                                          enableOutputDump,
                                                          uint32_t                                      outputDumpHaltCnt,
                                                          uint32_t                                      numOfTasks=1);
            virtual             ~Habana_singleStreamRunner();


            virtual void        loadSamplesToRam(uintptr_t clientData, const mlperf::QuerySampleIndex*, size_t);
            virtual void        unloadSamplesFromRam(uintptr_t clientData, const mlperf::QuerySampleIndex*, size_t);
            virtual void        queueQuery(uintptr_t ClientData, const mlperf::QuerySample*, size_t);
            virtual void        flushQueries();            
            virtual void        reportLatencyReport(uintptr_t ClientData, const int64_t*, size_t);
        protected:
            void  allocInputTensors();
            void  allocOutputTensors(bool                                       isEnforcedResBuf, 
                                     uint64_t                                   enforcedResBufByteSize,
                                     std::vector<std::unique_ptr<baseFunctor>>  &postProcFunctor,
                                     uint32_t                                   numOfThreads);
            virtual void  dumpRawOutputs(Habana_taskInfo &taskInfo, uint64_t halt_count);
            HABANA_benchStatus sutReportResults(Habana_taskInfo &);
            uint32_t                                    m_deviceId;                         //device ID give by synapse aquire function
            uint64_t                                    m_topologyId;                       // Habana loaded topology ID (issued for each recipe), received after load receipe
            uint64_t                                    m_numOfLoadedImages;                        // supported batchsize
            uint64_t                                    m_batchSize;                        // supported batchsize
            uint64_t                                    m_inputSizeInBytes;                 
            uint64_t                                    m_maxNumOfquerySample;
            uint64_t                                    m_numOfTasks;
            uint8_t                                     *m_queryBuffer;                     // query cache buffer (containing the image data of the query)
            std::shared_ptr<HABANA_imageCacheManager>   m_runnerImageCacheManager;
            std::vector<uint64_t>                       m_inputDims;        
            std::vector<Habana_taskInfo>                m_tasks;                          
            std::vector<std::unique_ptr<baseFunctor>>   m_transFunc;                        // results translation function and preprocess function.
            uint32_t                                    m_numOfInputs;                      //number of input tensors to the topology
            uint32_t                                    m_numOfOutputs;                     
            uint32_t                                    m_numOfInter;
            bool                                        m_enableOutputDump;
            uint32_t                                    m_outputDumpHaltCnt;
            char                                        m_inputName[1][ENQUEUE_TENSOR_NAME_MAX_SIZE];
            char                                        m_outputName[MAX_NUM_OUTPUT_TENSORS][ENQUEUE_TENSOR_NAME_MAX_SIZE];
            TensorMetadataInfo                          m_outputTensorInfo[MAX_NUM_OUTPUT_TENSORS];
    };



    //****************************************************************************************************************************************
    // Habana_multiStreamRunner - implements a runner dealing with multi stream scenario and also low latency server scenario when dealing
    // with queries with one sample per query.
    // the class can open up to 8 processing threads (defined by the constructor of this class)
    // the implemenation of the task scheduling to Habana HW is as follows:
    // the queue of a query samples to the Habana HW are done in the issueQuery functions while the wait for results and results reporting
    // to loadgen done in the processing thread. this ensures lowest latency and faster response.
    //****************************************************************************************************************************************

    class Habana_multiStreamRunner : public Habana_singleStreamRunner
    {
        public:

    //****************************************************************************************************************************************
    // Habana_multiStreamRunner performs class construction, the major part of the init is done in the base class constructor
    // Habana_singleStreamRunner class(all memory allocation), this constructor also sets up all needed threads and synchronization mechanisms
    // all parameters passed are similar to the one passed to the base class constructor but in this case user must pass numOfTasks since there
    // is no default value to this parameter.
    //****************************************************************************************************************************************
                            Habana_multiStreamRunner(const std::string                          &recipeName,
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
                                                     uint32_t                                   numOfTasks); 
                            virtual         ~Habana_multiStreamRunner();
                            virtual void    queueQuery(uintptr_t ClientData, const mlperf::QuerySample*, size_t);
        protected:
                            virtual void    process( Habana_taskInfo &taskInfo);
                            virtual void    queueSingleBatchQuery(uintptr_t ClientData, const mlperf::QuerySample *querySamples, size_t numOfquerySamples);
                            std::vector<ThrTask::ThreadedTask<std::function<void(void)>,std::function<void(void)>>>   m_enqueueThreads; 
                            
                             
    };
    class Habana_offlineRunner : public Habana_multiStreamRunner
    {
        public:
            Habana_offlineRunner(const std::string                          &recipeName,
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
                                 uint32_t                                   breakedBatchSize);
            virtual         ~Habana_offlineRunner();
            virtual void    queueQuery(uintptr_t ClientData, const mlperf::QuerySample*, size_t);
        private:
            uint32_t        m_breakBatchSize;
            
    };
    class Habana_serverRunner : public Habana_multiStreamRunner
    {
        public:
            Habana_serverRunner(const std::string                          &recipeName,
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
                                std::chrono::duration<int,std::micro>      procSlotDurationTime); 
            virtual         ~Habana_serverRunner();
            virtual void    queueQuery(uintptr_t ClientData, const mlperf::QuerySample*, size_t);
        private:
            void batchfier(int i,const mlperf::QuerySample* querySamples);
            void enqueueAndProc(int);
            void watchDog();
            std::atomic<bool>                           m_watchDogEnable;
            std::chrono::duration<int,std::micro>       m_procSlotDurationTime;
            std::vector<Hababan_batchifierData>         m_batchifierDataVect;
            std::thread                                 m_watchDogThread;
    };
}






#endif //HABANA_RUNNER