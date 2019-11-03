#ifndef HABANA_RUNNER_C_INTERFACE
#define HABANA_RUNNER_C_INTERFACE

#include <HABANA_runner.h>
#include <test_settings.h>
#include <string>

//************************************************************************************************************
//  the following code implements a simple API to call the different runners, this is needed since we need
// to pass specific runner function to be callback function withing the loadgen flow (and this can't be done 
// using class methods (loadgen has specific c like callable interface)
//************************************************************************************************************

namespace HABANA_bench
{
    
    #define SAFE_GUARD_LEN_RESNET50      1024
    #define SAFE_GUARD_LEN_SSD           64
    
    
//************************************************************************************************************
// runnerAllocRunner - the function identifies the runner to be used (single, multi, server or offline), passes
// allocates the runner to a unique_ptr and passes the parameters to the runner constructor.
//************************************************************************************************************
    void  runnerAllocRunner(mlperf::TestScenario                            runnerType,
                           const std::string                                recipeName,
                           const std::string                                imageDirePath,
                           const std::string                                imagListFile,
                           uint64_t                                         maxNumquerySamples,
                           uint64_t                                         numberOFImagesToLoad,
                           uint64_t                                         batchSize,
                           std::vector<std::unique_ptr<baseFunctor>>        &outputTransProc,  
                           bool                                             enableOutputDump,
                           uint32_t                                         outputDumpHaltCnt,
                           uint32_t                                         numOfTasks,
                           uint32_t                                         breakBatchSize,
                           std::chrono::duration<int,std::micro>            expectedImageProc,
                           std::chrono::duration<int,std::micro>            procSlotDurationTime,
                           bool                                             isEnforcedResBuf=false,
                           uint64_t                                         enforcedResBuf = 0);




    void        runnerLoadSamplesToRam(uintptr_t clientData, const mlperf::QuerySampleIndex*, size_t);
    void        runnerUnloadSamplesFromRam(uintptr_t clientData, const mlperf::QuerySampleIndex*, size_t);
    void        runnerQueueQuery(uintptr_t ClientData, const mlperf::QuerySample*, size_t);
    void        runnerFlushQuries();
    void        runnerReportLatencyReport(uintptr_t ClientData, const int64_t*, size_t);
    void        freeRunnerInterface();
} // namespace HABANA_bench



#endif //HABANA_RUNNER_C_INTERFACE