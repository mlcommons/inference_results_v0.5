#include <HABANA_runner_c_interface.h>
#include <string>
namespace HABANA_bench
{
     std::unique_ptr<Habana_singleStreamRunner> HABANA_runner;

    void runnerAllocRunner(mlperf::TestScenario                             runnerType,
                            const std::string                               recipeName,
                            const std::string                               imageDirePath,
                            const std::string                               imagListFile,
                            uint64_t                                        maxNumquerySamples,
                            uint64_t                                        numberOFImagesToLoad,
                            uint64_t                                        batchSize,
                            std::vector<std::unique_ptr<baseFunctor>>       &outputTransProc,  
                            bool                                            enableOutputDump,
                            uint32_t                                        outputDumpHaltCnt,
                            uint32_t                                        numOfTasks,
                            uint32_t                                        breakBatchSize,
                            std::chrono::duration<int,std::micro>           expectedImageProc,
                            std::chrono::duration<int,std::micro>           procSlotDurationTime,
                            bool                                            isEnforcedResBuf,
                            uint64_t                                        enforcedResBufSize)
    {
        switch (runnerType)
        {
        case mlperf::TestScenario::SingleStream:
            HABANA_runner.reset(new Habana_singleStreamRunner(recipeName,
                                                            imageDirePath,
                                                            imagListFile,
                                                            maxNumquerySamples,
                                                            numberOFImagesToLoad,
                                                            batchSize,
                                                            outputTransProc,
                                                            isEnforcedResBuf,
                                                            enforcedResBufSize,
                                                            enableOutputDump,
                                                            outputDumpHaltCnt));
            break;
        case mlperf::TestScenario::MultiStream:
            HABANA_runner.reset(new Habana_multiStreamRunner(recipeName,
                                                            imageDirePath,
                                                            imagListFile,
                                                            maxNumquerySamples,
                                                            numberOFImagesToLoad,
                                                            batchSize,
                                                            outputTransProc,
                                                            isEnforcedResBuf,
                                                            enforcedResBufSize,
                                                            enableOutputDump,
                                                            outputDumpHaltCnt,
                                                            numOfTasks));
            break;
        case mlperf::TestScenario::Server:
            if(isEnforcedResBuf)
            {
                HABANA_runner.reset(new Habana_multiStreamRunner(recipeName,
                                                                imageDirePath,
                                                                imagListFile,
                                                                maxNumquerySamples,
                                                                numberOFImagesToLoad,
                                                                batchSize,
                                                                outputTransProc,
                                                                isEnforcedResBuf,
                                                                enforcedResBufSize,
                                                                enableOutputDump,
                                                                outputDumpHaltCnt,
                                                                numOfTasks));
            }
            else
            {
                HABANA_runner.reset(new Habana_serverRunner(recipeName,
                                                                imageDirePath,
                                                                imagListFile,
                                                                maxNumquerySamples,
                                                                numberOFImagesToLoad,
                                                                batchSize,
                                                                outputTransProc,
                                                                isEnforcedResBuf,
                                                                enforcedResBufSize,
                                                                enableOutputDump,
                                                                outputDumpHaltCnt,
                                                                numOfTasks,
                                                                expectedImageProc,
                                                                procSlotDurationTime));        
            }
            break;
        case mlperf::TestScenario::Offline:
            HABANA_runner.reset(new Habana_offlineRunner(recipeName,
                                                             imageDirePath,
                                                             imagListFile,
                                                             maxNumquerySamples,
                                                             numberOFImagesToLoad,
                                                             batchSize,
                                                             outputTransProc,
                                                             isEnforcedResBuf,
                                                             enforcedResBufSize,
                                                             enableOutputDump,
                                                             outputDumpHaltCnt,
                                                             numOfTasks,
                                                             breakBatchSize));
            break;
        
        default:
            throw HABANA_benchException("HABANA_runner_interface - allocRunner - Wrong ruuner type");
        }
    }
    void        runnerLoadSamplesToRam(uintptr_t clientData, const mlperf::QuerySampleIndex* queryIndexBuf, size_t numOfIndexes)
    {
        HABANA_runner->loadSamplesToRam(clientData, queryIndexBuf, numOfIndexes);
    }
    void        runnerUnloadSamplesFromRam(uintptr_t clientData, const mlperf::QuerySampleIndex* queryIndexBuf, size_t numOfIndexes)
    {
        HABANA_runner->unloadSamplesFromRam(clientData, queryIndexBuf, numOfIndexes);
    }
    void        runnerQueueQuery(uintptr_t ClientData, const mlperf::QuerySample* sampleBuf, size_t numOfSamples)
    {
        HABANA_runner->queueQuery(ClientData,sampleBuf,numOfSamples);
    }
    void        runnerFlushQuries()
    {
        HABANA_runner->flushQueries();
    }
    void        runnerReportLatencyReport(uintptr_t ClientData, const int64_t* timingBuf, size_t numOfMeasurments)
    {
        HABANA_runner->reportLatencyReport(ClientData,timingBuf,numOfMeasurments);
    }
    
    void        freeRunnerInterface()
    {
        HABANA_runner.reset(nullptr);
    }

} // namespace HABANA_bench

