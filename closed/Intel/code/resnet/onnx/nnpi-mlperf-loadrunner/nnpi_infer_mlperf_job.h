#pragma once

#include "runner_lib_common_types.h"
#include "nnpi_network.h"
#include "infer_job.h"

#include <vector>
#include <set>
#include <memory>
#include <cstring>
#include <utility>

namespace runner_lib {

    template<class IOStream, class TimeMeasurer>
    class NNPIInferJobMlPerf : public InferJob {
    public:

        struct ValidationUserData_t {
            std::vector<typename IOStream::file_handle>& input_handle;
            std::vector<typename IOStream::file_handle>& output_handle;
            std::vector<std::vector<uint8_t>> output_data;
            std::vector<NamedResource> host_input;
            std::vector<NamedResource> host_output;
            uint32_t inf_id;
            uint32_t infer_index;
        };
        typedef ValidationUserData_t ValidationUserData;

        NNPIInferJobMlPerf(const NetworkConfig& cfg, uint32_t inference_id, const NNPIHostNet& nnpi_host_net, const NNPIDeviceNet& nnpi_dev_net, TimeMeasurer& time_measurer)
            : m_cfg(cfg)
            , m_inference_id(inference_id)
            , m_nnpi_host_net(&nnpi_host_net)
            , m_nnpi_dev_net(&nnpi_dev_net)
            , m_inferCmd(NNPI_INVALID_NNPIHANDLE)
            , m_time_measurer(time_measurer)
            , m_infer_index(0) {
                m_numInputs = nnpi_host_net.GetNumInputs();
                m_numOutputs = nnpi_host_net.GetNumOutputs();
                IOStream::Init();
        }

        virtual ~NNPIInferJobMlPerf() {

        }

        void Config(const RuntimeConfig& run_config) {
            m_run_validators = run_config.run_validators;
        }

        virtual void InitInfer() {
            auto& db = IOStream::GetImageDataDb();
            std::unique_lock<std::mutex> guard(initMutex);
            m_copyCommandsPerSample.resize(db.size());
            // create input resources and copy commands
            for (uint32_t i = 0; i < m_numInputs; i++) {
                //host
                NamedResource nr;
                THROW_IF_NEQ(nnpiHostNetworkGetInputDesc(m_nnpi_host_net->GetHandle(), i, nr.name, &nr.desc), NNPI_INF_NO_ERROR);
                //device
                THROW_IF_NEQ(nnpiDeviceResourceCreate(m_nnpi_dev_net->GetDeviceContext(), &nr.desc, &nr.handle), NNPI_INF_NO_ERROR);
                m_devInputs.push_back(nr);
            }

            ///offline_data.resize(db.size());
            {
                bool inResCmpl = false;
                if (db.size() == m_hostInputResourcesPerSample.size()) {
                    inResCmpl = true;
                } else {
                    m_hostInputResourcesPerSample.resize(db.size());
                }

                for (size_t ri = 0; ri < db.size(); ri++) {

                    if (db[ri].size() > 0) {
                        ///auto& res = offline_data[i];
                        auto &m_hostInputs = m_hostInputResourcesPerSample[ri];
                        ///m_nnpi_dev_net->m_device
                        auto &m_copyInputCmds = m_copyCommandsPerSample[ri];


                        m_input_handles.resize(m_numInputs);

                        // create input resources and copy commands
                        for (uint32_t i = 0; i < m_numInputs; i++) {
                            NamedResource nr;
                            if (!inResCmpl) {
                                //host
                                THROW_IF_NEQ(nnpiHostNetworkGetInputDesc(m_nnpi_host_net->GetHandle(), i, nr.name, &nr.desc), NNPI_INF_NO_ERROR);
                                THROW_IF_NEQ(nnpiHostResourceCreate(m_nnpi_host_net->GetAdapter(), &nr.desc, &nr.handle), NNPI_INF_NO_ERROR);
                                m_hostInputs.push_back(nr);
                            } else {
                                nr = m_hostInputs[i];
                            }
                            NNPIHostResource hInput = nr.handle; //save before overwriting with device handle;

                            //get device handle
                            nr = m_devInputs[i];

                            //copy
                            {
                                NNPICopyCommand copyInputCmd(NNPI_INVALID_NNPIHANDLE);
                                THROW_IF_NEQ(nnpiCopyCommandCreateHostToDevice(m_nnpi_dev_net->GetDeviceContext(), nr.handle, hInput, &copyInputCmd),
                                    NNPI_INF_NO_ERROR);
                                m_copyInputCmds.insert(copyInputCmd);
                            }
                            m_input_handles[i] = std::move(IOStream::Open(m_cfg.inputFiles.at(i)));
                            ////input_resource_ptr_map[i] = NULL;
                        }

                    }
                }

                // create output resources and copy commands
                for (uint32_t i = 0; i < m_numOutputs; i++) {
                    //host
                    NamedResource nr;
                    THROW_IF_NEQ(nnpiHostNetworkGetOutputDesc(m_nnpi_host_net->GetHandle(), i, nr.name, &nr.desc), NNPI_INF_NO_ERROR);
                    THROW_IF_NEQ(nnpiHostResourceCreate(m_nnpi_host_net->GetAdapter(), &nr.desc, &nr.handle), NNPI_INF_NO_ERROR);
                    m_hostOutputs.push_back(nr);
                    NNPIHostResource hOutput = nr.handle; //save before overwriting with device handle;

                    //device
                    THROW_IF_NEQ(nnpiDeviceResourceCreate(m_nnpi_dev_net->GetDeviceContext(), &nr.desc, &nr.handle), NNPI_INF_NO_ERROR);
                    m_devOutputs.push_back(nr);

                    //copy
                    NNPICopyCommand copyOutputCmd(NNPI_INVALID_NNPIHANDLE);
                    THROW_IF_NEQ(nnpiCopyCommandCreateDeviceToHost(m_nnpi_dev_net->GetDeviceContext(), hOutput, nr.handle, &copyOutputCmd), NNPI_INF_NO_ERROR);
                    m_copyOutputCmds.insert(copyOutputCmd);

                    ///output_resource_ptr_map[i] = NULL;

                    m_output_handles.emplace_back(IOStream::Open(m_cfg.outputFiles.at(i)));
                }

                m_outbuffer.resize(m_numOutputs);

                // create infer command
                auto inputHandles = std::make_unique<NNPIDeviceResource[]>(m_numInputs);
                auto outputHandles = std::make_unique<NNPIDeviceResource[]>(m_numOutputs);
                for (uint32_t i = 0; i < m_numInputs; i++) {
                    inputHandles[i] = m_devInputs.at(i).handle;
                }
                for (uint32_t i = 0; i < m_numOutputs; i++) {
                    outputHandles[i] = m_devOutputs.at(i).handle;
                }

                THROW_IF_NEQ(
                        nnpiInferCommandCreate(m_nnpi_dev_net->GetHandle(), inputHandles.get(), m_numInputs, outputHandles.get(), m_numOutputs, &m_inferCmd),
                        NNPI_INF_NO_ERROR);

                if (!inResCmpl) {
                    for (size_t ri = 0; ri < db.size(); ri++) {
                        if (db[ri].size() > 0) {
                            auto &m_hostInputs = m_hostInputResourcesPerSample[ri];

                            for (size_t id = 0; id < m_numInputs; id++) {

                                void *pInputData(NULL);
                                m_time_measurer.AddTimePoint(INPUT_RESOURCE_LOCK, this, id);
                                THROW_IF_NEQ(
                                        nnpiHostResourceLock(m_hostInputs.at(id).handle, NNPIInfLockType::NNPI_LOCK_FOR_WRITE, MaxLockWaitTime, &pInputData),
                                        NNPI_INF_NO_ERROR);
                                if (pInputData) {
                                    size_t inputSize = ResourceSizeInBytes(m_hostInputs.at(id).desc);

                                    m_time_measurer.AddTimePoint(INPUT_DATA_COPY, this, id);
                                    memcpy(pInputData, db[ri].data(), db[ri].size()*sizeof(db[ri][0]));
                                    ///db[ri].clear();
                                    ///IOStream::GetData(m_input_handles.at(id), pInputData, inputSize);
                                }
                                m_time_measurer.AddTimePoint(TimeMeasurer::measure_ctl::START, INPUT_RESOURCE_UNLOCK, this, id);
                                THROW_IF_NEQ(nnpiHostResourceUnlock(m_hostInputs.at(id).handle), NNPI_INF_NO_ERROR);
                                m_time_measurer.AddTimePoint(TimeMeasurer::measure_ctl::STOP, INPUT_RESOURCE_UNLOCK, this, id);
                            }
                        }
                    }

                }
            }
        }

        virtual void SetupInfer(int index) {
            m_infer_index = index;
            for (size_t id = 0; id < m_numInputs; id++)
            {
                IOStream::GetData(m_input_handles.at(id), nullptr, 0 );
            }
            ///std::cout << "Setup complete\n";
        }
        virtual void SendInfer() {
            auto& m_copyInputCmds = m_copyCommandsPerSample[m_input_handles[0].index];
///            auto& m_copyInputCmds = m_copyCommandsPerSample/*[dev]*/[m_input_handles[0].index];
//            auto& res = offline_data[m_input_handles[0].index];
//            auto& m_copyInputCmds = res.m_copyInputCmds;

            THROW_IF_EQ(m_copyInputCmds.size(), (size_t)0);
            // schedule copy input
            for (auto& copyHandle : m_copyInputCmds)
            {
                m_time_measurer.AddTimePoint(COPY_COMMAND, this);
                THROW_IF_NEQ(nnpiCopyCommandQueue(copyHandle, 0), NNPI_INF_NO_ERROR);
            }
            // schedule infer
            m_time_measurer.AddTimePoint(TimeMeasurer::measure_ctl::START, INFERENCE_COMMAND, this);
            THROW_IF_NEQ(nnpiInferCommandQueue(m_inferCmd, NULL), NNPI_INF_NO_ERROR);
            m_time_measurer.AddTimePoint(TimeMeasurer::measure_ctl::STOP, INFERENCE_COMMAND, this);

            // schedule copy output
            for (auto copyHandle : m_copyOutputCmds)
            {
                THROW_IF_NEQ(nnpiCopyCommandQueue(copyHandle, 0), NNPI_INF_NO_ERROR);
            }
        }

        virtual void CompleteInfer() {
//            auto& res = offline_data[m_input_handles[0].index];
//            auto& m_hostInputs = res.m_hostInputs;
            auto& m_hostInputs = m_hostInputResourcesPerSample[m_input_handles[0].index];

            for (size_t id = 0; id < m_numOutputs; id++)
            {
                void* pOutputData(nullptr);
                uint8_t* pRaw(nullptr);
                size_t outputSize(0);
                if (m_run_validators) {
                    outputSize = ResourceSizeInBytes(m_hostOutputs.at(id).desc);
                    auto& buf = m_outbuffer[id];
                    buf.resize(outputSize);
                    pRaw = buf.data();
                }

                m_time_measurer.AddTimePoint(OUTPUT_RESOURCE_LOCK, this);
                THROW_IF_NEQ(nnpiHostResourceLock(m_hostOutputs.at(id).handle, NNPIInfLockType::NNPI_LOCK_FOR_READ, MaxLockWaitTime, &pOutputData), NNPI_INF_NO_ERROR);

                if (m_run_validators && pOutputData) {
                    m_time_measurer.AddTimePoint(OUTPUT_DATA_COPY, this);
                    memcpy((void*) pRaw, pOutputData, outputSize);
                }

                m_time_measurer.AddTimePoint(TimeMeasurer::measure_ctl::START, OUTPUT_RESOURCE_UNLOCK, this);
                THROW_IF_NEQ(nnpiHostResourceUnlock(m_hostOutputs.at(id).handle), NNPI_INF_NO_ERROR);
                m_time_measurer.AddTimePoint(TimeMeasurer::measure_ctl::STOP, OUTPUT_RESOURCE_UNLOCK, this);
            }

            if (m_run_validators) {
                ValidationUserData val_data{m_input_handles, m_output_handles, m_outbuffer, m_hostInputs, m_hostOutputs, m_inference_id, m_infer_index};

                auto res = m_cfg.validator->validate(m_cfg, &val_data);
                (void)res;
            }
        }

        virtual void DestroyInfer() {

            {
                //std::lock_guard<std::mutex> guard(initMutex);
                for (auto &m_copyInputCmds : m_copyCommandsPerSample) {
                    // destroy commands
                    for (auto &copyHandle : m_copyInputCmds) {
                        THROW_IF_NEQ(nnpiCopyCommandDestroy(copyHandle), NNPI_INF_NO_ERROR);
                    }
                    m_copyInputCmds.clear();
                }
                m_copyCommandsPerSample.clear();
            }
            for (auto &copyHandle : m_copyOutputCmds) {
                THROW_IF_NEQ(nnpiCopyCommandDestroy(copyHandle), NNPI_INF_NO_ERROR);
            }
            m_copyOutputCmds.clear();

            THROW_IF_NEQ(nnpiInferCommandDestroy(m_inferCmd), NNPI_INF_NO_ERROR);
            m_inferCmd = NNPI_INVALID_NNPIHANDLE;

            // destroy device resources
            for (auto nr : m_devInputs)
            {
                THROW_IF_NEQ(nnpiDeviceResourceDestroy(nr.handle), NNPI_INF_NO_ERROR);
            }
            m_devInputs.clear();

            for (auto nr : m_devOutputs)
            {
                THROW_IF_NEQ(nnpiDeviceResourceDestroy(nr.handle), NNPI_INF_NO_ERROR);
            }
            m_devOutputs.clear();

            {
                std::lock_guard<std::mutex> guard(initMutex);
                for(auto& m_hostInputs:m_hostInputResourcesPerSample) {
                    // destroy host resources
                    for (auto nr : m_hostInputs)
                    {
                        THROW_IF_NEQ(nnpiHostResourceDestroy(nr.handle), NNPI_INF_NO_ERROR);
                    }
                    m_hostInputs.clear();
                }
                m_hostInputResourcesPerSample.clear();
            }
            for (auto nr : m_hostOutputs) {
                THROW_IF_NEQ(nnpiHostResourceDestroy(nr.handle), NNPI_INF_NO_ERROR);
            }
            m_hostOutputs.clear();

        }
    private:
        const NetworkConfig& m_cfg;
        uint32_t m_inference_id;
        const NNPIHostNet* m_nnpi_host_net;
        const NNPIDeviceNet* m_nnpi_dev_net;
        uint32_t m_numInputs;
        uint32_t m_numOutputs;
        bool m_run_validators = true;

        std::vector<NamedResource> m_devInputs, m_devOutputs;

        static std::vector<std::vector<NamedResource>> m_hostInputResourcesPerSample;
        static std::mutex initMutex;

        //static std::map<void*, std::vector<std::set<NNPICopyCommand>>> m_copyCommandsPerDevicePerSample;
	std::vector<std::set<NNPICopyCommand>> m_copyCommandsPerSample;
        //std::vector<std::set<NNPICopyCommand>> m_copyCommandsPerSample;

        std::vector<NamedResource> m_hostOutputs;
        std::set<NNPICopyCommand> m_copyOutputCmds;
        NNPIInferCommand m_inferCmd;
        std::vector<typename IOStream::file_handle> m_input_handles;
        std::vector<typename IOStream::file_handle> m_output_handles;
        TimeMeasurer& m_time_measurer;
        uint32_t m_infer_index;
        std::vector<std::vector<uint8_t>> m_outbuffer;
    };

}
