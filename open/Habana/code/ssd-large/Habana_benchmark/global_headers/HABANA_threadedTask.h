#ifndef __THREADED_TASK__
#define __THREADED_TASK__
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
namespace ThrTask
{
    enum class WorkStatus
{
    IDLE,
    WORKING,
    STOP_EXECUTION
};
    template <class processFunctor, class postProcFunctor>
    class ThreadedTask
    {
        public:
            ThreadedTask(const ThreadedTask& other) = delete;
            ThreadedTask& operator=(const ThreadedTask & other) = delete;
            ThreadedTask() : m_workStatus(WorkStatus::IDLE), m_lockingMutex(new std::mutex), m_signalingCond(new std::condition_variable)
            {
                processFunctor  tempProc;
                postProcFunctor tempPost;
                m_task      = tempProc;
                m_postProc  = tempPost;
                m_executionThread = std::thread(&ThreadedTask<processFunctor, postProcFunctor>::executeTasks,this);
            }
            ThreadedTask(processFunctor &initProcFunctor, postProcFunctor &initPostProcFunctor) : m_workStatus(WorkStatus::IDLE), m_lockingMutex(new std::mutex), m_signalingCond(new std::condition_variable)
            {
                m_task      = initProcFunctor;
                m_postProc  = initPostProcFunctor;
                m_executionThread = std::thread(&ThreadedTask<processFunctor, postProcFunctor>::executeTasks,this);
            }
            ThreadedTask(ThreadedTask &&other)
            {
                while(other.isBusy());
                m_workStatus        = other.m_workStatus;
                m_lockingMutex      = std::move(other.m_lockingMutex);
                m_signalingCond     = std::move(other.m_signalingCond);
                m_executionThread   = std::move(other.m_executionThread);
                m_postProc          = other.m_postProc;
                m_task              = other.m_task;
                other.m_workStatus = WorkStatus::STOP_EXECUTION;
                
            }
            ThreadedTask& operator=(ThreadedTask &&other)
            {
                while(other.isBusy());
                m_workStatus        = other.m_workStatus;
                m_lockingMutex      = std::move(other.m_lockingMutex);
                m_signalingCond     = std::move(other.m_signalingCond);
                m_executionThread   = std::move(other.m_executionThread);
                m_postProc          = other.m_postProc;
                m_task              = other.m_task;
                return *this;
            }
            ~ThreadedTask()
            {
                killThread();
            }
            void killThread()
            {
                if (m_workStatus != WorkStatus::STOP_EXECUTION)
                {
                    std::lock_guard<std::mutex> lock(*m_lockingMutex);
                    m_workStatus = WorkStatus::STOP_EXECUTION;
                }
                m_signalingCond->notify_one();
                if (m_executionThread.joinable())
                    m_executionThread.join();
            }
            bool isBusy()
            {
                return ((m_workStatus == WorkStatus::WORKING) || (m_workStatus == WorkStatus::STOP_EXECUTION));
            }
            void sendTask(processFunctor &newTask, postProcFunctor &newPostProc)
            {
                {
                    std::lock_guard<std::mutex> lock(*m_lockingMutex);
                    m_workStatus = WorkStatus::WORKING;
                    m_task = newTask;
                    m_postProc = newPostProc;
                }
                
                m_signalingCond->notify_one();
            }
            void sendTaskCpy(processFunctor newTask, postProcFunctor newPostProc)
            {
                {
                    std::lock_guard<std::mutex> lock(*m_lockingMutex);
                    m_workStatus = WorkStatus::WORKING;
                    m_task = newTask;
                    m_postProc = newPostProc;
                }
                
                m_signalingCond->notify_one();
            }
            
            void sendTask()
            {
                {
                    std::lock_guard<std::mutex> lock(*m_lockingMutex);
                    m_workStatus = WorkStatus::WORKING;
                }
                m_signalingCond->notify_one();
            }

            void executeTasks(void)
            {
                while(1)
                {
                    std::unique_lock<std::mutex> lock(*m_lockingMutex);
                    // wait on the condition variable for a task to come in, the this thread dictates != IDLE
                    m_signalingCond->wait(lock, [&]() {return (m_workStatus != WorkStatus::IDLE);});
                    if (m_workStatus == WorkStatus::STOP_EXECUTION)
                        break;
                    m_task();
                    m_postProc();
                    m_workStatus = WorkStatus::IDLE;                
                }
            }


        private:
            volatile WorkStatus                        m_workStatus;
            processFunctor                             m_task;
            postProcFunctor                            m_postProc;
            std::thread                                m_executionThread;
            std::unique_ptr<std::mutex>                m_lockingMutex;     // mutex object to be used with the condition variable.
                                                                           // that is need to lock the mutex before changing the changing the condition variable
            std::unique_ptr<std::condition_variable>   m_signalingCond;    // signalling condition variable
    };
}

#endif