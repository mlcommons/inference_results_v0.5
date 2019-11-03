#ifndef LOADRUN_MULTI_INSTANCE_H_
#define LOADRUN_MULTI_INSTANCE_H_

#include "query_sample.h"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <chrono>

// use longer one between mlperf::QuerySampleIndex and mlperf::ResponseId
typedef mlperf::ResponseId Msg_Size;

// 1 query has 1 img and 1 response id, 
// mlperf requires at least 24k queries  
// so the upper limit can be 25K * 2, and *x for faster machine for min duration constraints
// for super fast engine, specify larger buffer size with --msg_buffer_size
const Msg_Size MAX_MSG_COUNT = 800000;
const Msg_Size CMD_START = 2;
const Msg_Size CMD_STOP = 1;
const Msg_Size CMD_TEST_END = 3;
const Msg_Size CMD_QUEUE_QUERY = 0xFFFFFFFF;
const Msg_Size CMD_SYNC_OFFSET = 1; // sync the start/stop of netrun instances
const Msg_Size CMD_LOADRUN_MAIN_OFFSET = 3; // stop loadrun main process
const Msg_Size CMD_CHECK_THREAD_OFFSET = 5; // stop check response thread
const Msg_Size CMD_TEST_END_OFFSET = 7; // notify FlushQuery signal
const Msg_Size CMD_SCHEDULE_LOCAL_BATCH_FIRST = 10; //notify client to handle local batcheds together
const Msg_Size STATUS_DONE = 12; // Cient notify query is completed to server
const Msg_Size CMD_LOAD_SAMPLES = 16; // notify client to load samples
const Msg_Size STATUS_LOAD_SAMPLES_DONE = 18; // notify server load samples is done
const Msg_Size STATUS_LOAD_SAMPLES_OFFSET = 11; // notify server load samples is done

namespace loadrun {

struct LoadrunSettings {
  // reserve different queue size and batch size for scheduling experiment
  Msg_Size loadrun_queue_size = 1;
  Msg_Size batch_size = 1;

  bool is_offline = false;
  bool include_accuracy = true;
  Msg_Size offline_samples = std::numeric_limits<Msg_Size>::max();
};

} // name space loadrun

struct trace_queue
{
   boost::interprocess::interprocess_mutex      loadsample_mutex;
   boost::interprocess::interprocess_mutex      instance_start_mutex; 
   boost::interprocess::interprocess_mutex      instance_done_mutex;   
   boost::interprocess::interprocess_condition  cond_loadsample_done;
   boost::interprocess::interprocess_condition  cond_instance_start;   
   boost::interprocess::interprocess_condition  cond_instance_done;
   size_t instance_done_counter = 0;
   size_t instance_loadsample_counter = 0;
};

struct local_timer
{
  std::chrono::time_point<std::chrono::high_resolution_clock> local_start;
  std::chrono::time_point<std::chrono::high_resolution_clock> local_end;	
  std::chrono::duration<double, std::milli> local_duration;

  void start(){
#ifdef PROFILE    
    local_start = std::chrono::high_resolution_clock::now();
#endif    
  }
  
  void end(){
#ifdef PROFILE     
    local_end = std::chrono::high_resolution_clock::now();
    local_duration += local_end - local_start;
#endif     
  }
  
  // return ms
  double get_ms_duration(){
#ifdef PROFILE     
    return local_duration.count();
#else
    return 1;    
#endif     
  }
};

#endif  // LOADRUN_MULTI_INSTANCE_H_