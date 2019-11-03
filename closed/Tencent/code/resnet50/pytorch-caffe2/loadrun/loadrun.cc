#include <string> 
#include <vector>

// loadgen integration
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"
#include <thread>
#include <chrono>
#include <iostream>
#include "inferencer.h"

// multi instances support
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/detail/config_begin.hpp>
#include <cstdio>
#include "multi_instance.h"
#include <limits>

// loadgen integration
mlperf::TestSettings mlperf_args;
static size_t inference_samples = 0;
static size_t number_response_samples = 0;
std::unique_ptr<loadrun::Inferencer> sut_classifier;
local_timer sort_timer;

// multi instances support
std::vector<std::string> ins_names;
std::string server_name("");
std::vector<Msg_Size *> pos_c_cmd_vector;
std::vector<Msg_Size *> pos_c_rsp_vector;
std::vector<Msg_Size *> pos_c_status_vector;
std::vector<Msg_Size *> pos_c_sync_vector;
Msg_Size *pos_server_cmd;
bool stop_process = false;
Msg_Size instance_index = 0;
bool run_as_instance = false;
bool interactive = false;
bool offline_single_response = false;
// sleep for a while after check each instances' response
bool loadrun_sleep_after_each_response_check = true; 
// sort samples before dispatch
bool sort_samples = false;
// handle queries number indivisable by batch size
bool handle_flush_queries = true;
bool loadgen_test_end = false;
// offline batch dispatch for continuous samples in ram
bool schedule_local_batch_first = false;
size_t number_instances;
size_t quotient_batch;
size_t remain_samples;  
size_t instance_batch;
size_t remain_batch;
size_t total_batch;
bool use_mlperf_config_file = false;
std::string mlperf_config_file_name("");
bool use_user_config_file = false;
std::string user_config_file_name("");
std::string model_name("");
std::string scenario_name("");
size_t instance_loadgen_makeup = 0;
trace_queue * sync_cond;
bool sync_by_cond = false;

local_timer ben_timer;
local_timer query_complete_timer;
local_timer issue_query_timer;

loadrun::LoadrunSettings loadrun_args;

extern std::unique_ptr<loadrun::Inferencer> get_inferencer();

int compare_sample(const void* a, const void* b){
  mlperf::QuerySampleIndex arg1;
  mlperf::QuerySampleIndex arg2;

  arg1 = static_cast<const mlperf::QuerySample*>(a)->index;
  arg2 = static_cast<const mlperf::QuerySample*>(b)->index;

  if(arg1 < arg2) return -1;
  if(arg1 > arg2) return 1;
  return 0;
}

using namespace boost::interprocess;

void IssueQuery(mlperf::c::ClientData client_data, const mlperf::QuerySample* samples, size_t samples_size) {    
  // run loadrun and inference as a single process
  if(!run_as_instance){    
    // batch size 1, 1 sample per query
    sut_classifier->prepare_batch(samples->index);
    sut_classifier->run(0, true);
    if(loadrun_args.include_accuracy){
      std::vector<int> results = sut_classifier->get_labels(0, true);           
      mlperf::QuerySampleResponse response{samples->id, 
                                           (uintptr_t)(&results[0]), 
                                           sizeof(Msg_Size)};    
      mlperf::QuerySamplesComplete(&response, samples_size);                                                 
    }else{
      mlperf::QuerySampleResponse response{samples->id, 0, 0};
      mlperf::QuerySamplesComplete(&response, samples_size);      
    }  
    inference_samples++;
    return;    
  }  

  // multi instances support
  issue_query_timer.start();
  loadrun_args.offline_samples = samples_size;

  // assume dispatch to client by batch size
  // we may try different queue size with batch size, remove it at that time
  assert(loadrun_args.loadrun_queue_size == loadrun_args.batch_size);    

  Msg_Size dispatch_token = 0;
  const mlperf::QuerySample* end = samples + samples_size;
  const mlperf::QuerySample* p = samples;
 
  if (sort_samples){
    sort_timer.start();
    std::qsort(const_cast<mlperf::QuerySample*>(samples), samples_size, sizeof(mlperf::QuerySample), compare_sample);
    sort_timer.end();
  }

  size_t instance_loadgen_makeup_counter = instance_loadgen_makeup * (pos_c_cmd_vector.size()-1);  
  if(!schedule_local_batch_first){    
    // select inference instance by round robin with sample
    while (p < end) { 
      if (inference_samples < instance_loadgen_makeup_counter){
        // skip instance_loadgen_makeup samples for instance 1, if it's specified with loadgen in same instance
        dispatch_token = (inference_samples/loadrun_args.loadrun_queue_size)%(pos_c_cmd_vector.size()-1);            
        dispatch_token++;
      }else{      
        dispatch_token = (inference_samples/loadrun_args.loadrun_queue_size)%pos_c_cmd_vector.size();    
      }

      // send batch_size*(img index and response id) to client cmd buffer
      // +1 for index 0 img, since we use 0 to mark no command
      // move to next cmd holder position
      Msg_Size queue_size = 0;
      while((queue_size < loadrun_args.loadrun_queue_size) && (p < end)){ 
        // send response id to client cmd buffer
        *(pos_c_cmd_vector[dispatch_token]) = p->id;       
        (pos_c_cmd_vector[dispatch_token])++; 
        
        // send image index to client cmd buffer 
        *(pos_c_cmd_vector[dispatch_token]) = p->index + 1;  
        (pos_c_cmd_vector[dispatch_token])++;

        p++;
        inference_samples++;
        queue_size++;
      }
    }
    
    issue_query_timer.end();    

    if (sync_by_cond){
      // good for offline scenario
      // notify all instances start to work
      scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);   
      sync_cond->cond_instance_start.notify_all(); 
      lock_instance_start.unlock();
      
      // block self and wait for instance completes all samples inferencing
      scoped_lock<interprocess_mutex> lock(sync_cond->instance_done_mutex);    
      sync_cond->cond_instance_done.wait(lock);    
      lock.unlock();

      // the instance may be called several times, reset counter for next invoke
      sync_cond->instance_done_counter = ins_names.size();
            
      // Count responses until all done 
      std::vector<mlperf::QuerySampleResponse> responses;      
      for (size_t index = 0; index<pos_c_rsp_vector.size(); index++){
        while (*(pos_c_rsp_vector[index]) != 0){
          if(loadrun_args.include_accuracy){             
            mlperf::QuerySampleResponse response{*(pos_c_rsp_vector[index]), 
                                                (uintptr_t)(pos_c_rsp_vector[index]+1), 
                                                sizeof(Msg_Size)};
            responses.push_back(response);
            (pos_c_rsp_vector[index])+=2; 
          }else{
            mlperf::QuerySampleResponse response{*(pos_c_rsp_vector[index]), 0, 0};
            responses.push_back(response);              
            (pos_c_rsp_vector[index])++;
          }
          number_response_samples++;
        }
      }

      query_complete_timer.start(); 
      mlperf::QuerySamplesComplete(&responses[0], loadrun_args.offline_samples);          
      query_complete_timer.end();       
      
      responses.clear();                 
    }
  }else{
    // divide whole offline samples by instance and batch size with high locality and dispatch
    // Buffer cmd format
    //    CMD_SCHEDULE_LOCAL_BATCH_FIRST
    //    length
    //    start batch index
    quotient_batch = samples_size/loadrun_args.batch_size;
    remain_samples = samples_size%loadrun_args.batch_size;  
    total_batch = quotient_batch + (remain_samples>0?1:0);
    instance_batch = total_batch/number_instances;
    remain_batch = total_batch%number_instances; 
    // -1 for instacne start with 0
    size_t instance_index_with_last_batch = (remain_batch > 0)?(remain_batch - 1):(number_instances - 1);
    size_t dispatched_batches = 0;
    Msg_Size length = 0;
    for (size_t index_instance = 0; index_instance < number_instances; index_instance++){
      // CMD_SCHEDULE_LOCAL_BATCH_FIRST
      *(pos_c_cmd_vector[index_instance]) = CMD_SCHEDULE_LOCAL_BATCH_FIRST;
      (pos_c_cmd_vector[index_instance])++;
      
      // how many batches for each instance
      if (index_instance <= instance_index_with_last_batch){
        length = remain_batch>0?(instance_batch + 1):instance_batch;
      }else{
        length = instance_batch;
      }
      *(pos_c_cmd_vector[index_instance]) = length;
      (pos_c_cmd_vector[index_instance])++;      
       
      // start batch, index start from 0
      *(pos_c_cmd_vector[index_instance]) = dispatched_batches;
      (pos_c_cmd_vector[index_instance])++; 
      dispatched_batches+=length;
    }
    issue_query_timer.end();
    
    if(sync_by_cond){
      // synv by interprocess conditional variable
      // notify all instances start to work
      scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);   
      sync_cond->cond_instance_start.notify_all(); 
      lock_instance_start.unlock();
      
      // block self and wait for instance completes all samples inferencing
      scoped_lock<interprocess_mutex> lock(sync_cond->instance_done_mutex);    
      sync_cond->cond_instance_done.wait(lock);    
      lock.unlock();

      // the loadsample may be called several times, reset counter for next invoke
      sync_cond->instance_done_counter = ins_names.size();
    }else{
      // sync by poll
      // wait for all instances done      
      size_t instance_done = 0;
      while(true){
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        for (size_t index = 0; index<pos_c_status_vector.size(); index++){       
          if((*pos_c_status_vector[index])==STATUS_DONE){
            instance_done++;
          }      
        }
        if (instance_done == pos_c_status_vector.size()){
          for (size_t index = 0; index<pos_c_status_vector.size(); index++){
            // there may be more than 1 queries generated
            *(pos_c_status_vector[index]) = 0;
          }
          break;
        }else{
          instance_done = 0;
        }
      }
    }

    // check response
    remain_batch = total_batch%number_instances; 
    Msg_Size response_samples = 0;
    size_t instance_samples = 0;
    std::vector<mlperf::QuerySampleResponse> responses;
    for (size_t index = 0; index<pos_c_rsp_vector.size(); index++){
      // there's inference result in the buffer
      if (index <= instance_index_with_last_batch){
        instance_samples = (remain_batch>0?(instance_batch + 1):instance_batch) * loadrun_args.batch_size;
      }else if (index != (pos_c_rsp_vector.size() - 1)){
        instance_samples = instance_batch * loadrun_args.batch_size;
      }else{
        // the last batch, disgard supplement samples for lastest batch
        instance_samples = (instance_batch - 1) * loadrun_args.batch_size;
        instance_samples += (remain_samples>0?remain_samples:loadrun_args.batch_size);
      }

      size_t index_result;      
      for (index_result = 0; index_result < instance_samples; index_result++ ){
        if(loadrun_args.include_accuracy){ 
          mlperf::QuerySampleResponse response{(samples + response_samples + index_result)->id, 
                                              (uintptr_t)(pos_c_rsp_vector[index] + index_result), 
                                              sizeof(Msg_Size)};
          responses.push_back(response); 
        }else{
          mlperf::QuerySampleResponse response{(samples + response_samples + index_result)->id, 0, 0};            
          responses.push_back(response);
        }
      }

      response_samples += index_result;
      pos_c_rsp_vector[index] += loadrun_args.batch_size + 1;
    }
    inference_samples += response_samples;

    query_complete_timer.start();
    mlperf::QuerySamplesComplete(&responses[0], samples_size);     
    query_complete_timer.end();         
    
    responses.clear();    
  }

  return;
}

void check_response_by_thread(std::vector<Msg_Size *> rsp_position_vector){  
  std::vector<mlperf::QuerySampleResponse> responses;
  while(true){    
    if ((*pos_server_cmd) != CMD_STOP){
      if (!loadrun_args.is_offline || !offline_single_response) {    
        // server scenario 
        // or response 1 sample immediately in offline scenario
        for (size_t index = 0; index<rsp_position_vector.size(); index++){      
          while ((*(rsp_position_vector[index]) != 0) &&
                 (*(rsp_position_vector[index] + 1) != 0)) {            
            // get a response from client, notify loadgen issue is completed
            // move to next response holder position
            if(loadrun_args.include_accuracy){   
              // handle inference result 0
              *(rsp_position_vector[index] + 1) -= 1;

              mlperf::QuerySampleResponse response{*(rsp_position_vector[index]), 
                                                   (uintptr_t)(rsp_position_vector[index] + 1), 
                                                   sizeof(Msg_Size)};
              query_complete_timer.start();              
              mlperf::QuerySamplesComplete(&response, 1);
              query_complete_timer.end();              
              (rsp_position_vector[index])+=2;                                                   
            }else{
              mlperf::QuerySampleResponse response{*(rsp_position_vector[index]), 0, 0};  

              query_complete_timer.start();
              mlperf::QuerySamplesComplete(&response, 1);
              query_complete_timer.end();

              (rsp_position_vector[index])++;                   
            }				            
            number_response_samples++;             
          }
        }
      }else{
        // Offline scenario, count responses until all done 
        for (size_t index = 0; index<rsp_position_vector.size(); index++){
          while (*(rsp_position_vector[index]) != 0 && 
                  (responses.size() < loadrun_args.offline_samples)){
            if(loadrun_args.include_accuracy){ 
              mlperf::QuerySampleResponse response{*(rsp_position_vector[index]), 
                                                   (uintptr_t)(rsp_position_vector[index]+1), 
                                                   sizeof(Msg_Size)};
              responses.push_back(response);
              (rsp_position_vector[index])+=2; 
            }else{
              mlperf::QuerySampleResponse response{*(rsp_position_vector[index]), 0, 0};
              responses.push_back(response);              
              (rsp_position_vector[index])++;
            }
            number_response_samples++;
          }

          if (responses.size() == loadrun_args.offline_samples){
            query_complete_timer.start();  
            mlperf::QuerySamplesComplete(&responses[0], loadrun_args.offline_samples);          
            query_complete_timer.end();            
            
            responses.clear();           
            // break for loop of all other clients' response check 
            break;
          }
        }
      }
    }
    
    if (*(pos_server_cmd + CMD_CHECK_THREAD_OFFSET) == CMD_STOP){
      named_mutex named_mtx{open_or_create, ("mtx" + server_name).c_str()};
      named_mtx.lock();      
      std::cout << "loadrun checks response of " <<  number_response_samples << " samples\n";	  
      *(pos_server_cmd + CMD_LOADRUN_MAIN_OFFSET) = CMD_STOP; 
      named_mtx.unlock();
      return;
    }
    
    if (loadrun_sleep_after_each_response_check){
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
  }  
  return;
}

void ReportLatencyResults(mlperf::c::ClientData client_data, const int64_t*, size_t sample_size) {
  return;
}

void LoadSamplesToRam(mlperf::c::ClientData client_data, const mlperf::QuerySampleIndex* samples, size_t sample_size) {
  if(schedule_local_batch_first){   
    for (size_t index_instance = 0; index_instance<pos_c_rsp_vector.size(); index_instance++){
      // buffer format
      // cmd
      // size    
      Msg_Size *pos_cmd = pos_c_cmd_vector[index_instance];
      (pos_c_cmd_vector[index_instance])++; 
      *(pos_c_cmd_vector[index_instance]) = sample_size;
      (pos_c_cmd_vector[index_instance])++;       

      for (size_t index = 0; index < sample_size; index++){        
        *(pos_c_cmd_vector[index_instance]) = *(samples + index);       
        (pos_c_cmd_vector[index_instance])++;  
      }

      (*pos_cmd) = CMD_LOAD_SAMPLES;
    }
    
    if(sync_by_cond){
      // notify all instances start to work
      scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);   
      sync_cond->cond_instance_start.notify_all(); 
      lock_instance_start.unlock();
      
      // block self and wait for load sample of all instances are done
      scoped_lock<interprocess_mutex> lock(sync_cond->loadsample_mutex);
      sync_cond->cond_loadsample_done.wait(lock);

      // the loadsample may be called several times, e.g., in SubmissionRun mode.
      // reset counter for next invoke
      sync_cond->instance_loadsample_counter = ins_names.size();  
    }else{
      // wait for all instances done      
      size_t instance_done = 0;
      while(true){
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        for (size_t index = 0; index<pos_c_status_vector.size(); index++){       
          if((*(pos_c_status_vector[index] + STATUS_LOAD_SAMPLES_OFFSET))==STATUS_LOAD_SAMPLES_DONE){
            instance_done++;
          }      
        }
        if (instance_done == pos_c_status_vector.size()){
          for (size_t index = 0; index<pos_c_status_vector.size(); index++){
            // there may be more than 1 queries generated
            *(pos_c_status_vector[index] + STATUS_LOAD_SAMPLES_OFFSET) = 0;
          }
          break;
        }else{
          instance_done = 0;
        }
      }
    }
  }
  return;
}

void UnloadSamplesFromRam(mlperf::c::ClientData client_data, const mlperf::QuerySampleIndex* samples, size_t sample_size) {
  return;
}

void FlushQueries() {
  if(run_as_instance && handle_flush_queries){
    *(pos_server_cmd + CMD_TEST_END_OFFSET) = CMD_TEST_END;    
  }
  return;
}

uint64_t constexpr mix(char m, uint64_t s) {
  return ((s<<7) + ~(s>>3)) + ~m;
}
 
uint64_t constexpr hash(const char * m) {
  return (*m) ? mix(*m,hash(m+1)) : 0;
}

int main(int argc, char **argv) {  
  size_t performance_samples = 0;
  size_t total_samples = 0;
  size_t msg_buffer_size = MAX_MSG_COUNT * sizeof(Msg_Size);

  // move framework related args to new array for initialization  
  std::vector<char *> inferencer_argv;
  int inferencer_argc = 0;
  for(int index = 0; index < argc; index++ ){
    std::string arg(argv[index]);      
    switch(hash(argv[index]) ){
      case hash("--performance_samples"):
        performance_samples = std::stoi(argv[++index]);
        break;  
      case hash("--total_samples"):
        total_samples = std::stoi(argv[++index]);        
        break;
      case hash("--offline_expected_qps"):
        mlperf_args.offline_expected_qps = std::stoi(argv[++index]);         
        break;        
      case hash("--min_query_count"):
        mlperf_args.min_query_count = std::stoi(argv[++index]);        
        break;     		
      case hash("--max_query_count"):
        mlperf_args.max_query_count = std::stoi(argv[++index]);         
        break;
      case hash("--min_duration_ms"):
        mlperf_args.min_duration_ms = std::stoi(argv[++index]);         
        break;
      case hash("--max_duration_ms"):
        mlperf_args.max_duration_ms = std::stoi(argv[++index]);         
        break;        
      case hash("--server_target_qps"):
        mlperf_args.server_target_qps = std::stoi(argv[++index]);         
        break;   
      case hash("--qsl_rng_seed"):
        mlperf_args.qsl_rng_seed = std::stoi(argv[++index]);         
        break;  
      case hash("--sample_index_rng_seed"):
        mlperf_args.sample_index_rng_seed = std::stoi(argv[++index]);         
        break;       
      case hash("--schedule_rng_seed"):
        mlperf_args.schedule_rng_seed = std::stoi(argv[++index]);         
        break;                  
      case hash("--server_coalesce_queries"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          mlperf_args.server_coalesce_queries = true;
        }else{
          mlperf_args.server_coalesce_queries = false;     
        }               
        break;     
      }
      case hash("--server_target_latency_ns"):
        mlperf_args.server_target_latency_ns = std::stol(argv[++index]);   
        break;   
      case hash("--single_stream_expected_latency_ns"):
        mlperf_args.single_stream_expected_latency_ns = std::stol(argv[++index]);   
        break;                                      
      case hash("--mode"):{
        std::string mode(argv[++index]);
        if (mode.compare("PerformanceOnly") == 0){
          mlperf_args.mode = mlperf::TestMode::PerformanceOnly;
        }else if (mode.compare("AccuracyOnly") == 0){
          mlperf_args.mode = mlperf::TestMode::AccuracyOnly;
        }else if (mode.compare("SubmissionRun") == 0){
          mlperf_args.mode = mlperf::TestMode::SubmissionRun;
        }else if (mode.compare("FindPeakPerformance") == 0){
          mlperf_args.mode = mlperf::TestMode::FindPeakPerformance;
        }else{
          std::cout << "Uknown mode\n";
          std::exit(EXIT_FAILURE);      
        }          
        break;
      }
      case hash("--loadrun_queue_size"):
        loadrun_args.loadrun_queue_size = std::stoi(argv[++index]);         
        break;              
      case hash("--scenario"):{
        std::string temp(argv[++index]);
        scenario_name = temp;
        if (scenario_name.compare("Offline") == 0){
          mlperf_args.scenario = mlperf::TestScenario::Offline;
          loadrun_args.is_offline = true;
        }else if (scenario_name.compare("Server") == 0){
          mlperf_args.scenario = mlperf::TestScenario::Server;          
        }else if (scenario_name.compare("SingleStream") == 0){
          mlperf_args.scenario = mlperf::TestScenario::SingleStream;          
        }else if (scenario_name.compare("MultiStream") == 0){
          mlperf_args.scenario = mlperf::TestScenario::MultiStream;          
        }else {
          std::cout << "Unknown scenario\n";
          std::exit(EXIT_FAILURE);
        }                   
        break;         
      }
      case hash("--enable_spec_overrides"):{
        // skip it for backward compitability of previous script with old style loadgen
        index++;                 
        break;                 
      }
      case hash("--instance"):{
        std::string temp = std::string(argv[++index]);
        ins_names.push_back(temp);
        run_as_instance = true;
        }
        break;  
      case hash("--server"):{
        server_name = std::string(argv[++index]);        
        run_as_instance = true;
        }
        break;  
      case hash("--interactive"):{
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          interactive = true;
        }else{
          interactive = false;     
        }               
        break;                 
      }          
      case hash("--offline_single_response"):{
        // when true, put all samples in one query's response
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          offline_single_response = true;
        }else{
          offline_single_response = false;     
        }               
        break;                 
      }              
      case hash("--loadrun_sleep_after_each_response_check"):{
        // when true, put all samples in one query's response
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          loadrun_sleep_after_each_response_check = true;
        }else{
          loadrun_sleep_after_each_response_check = false;     
        }               
        break;                 
      }      
      case hash("--flush_queries"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          handle_flush_queries = true;
        }else{
          handle_flush_queries = false;     
        }               
        break;     
      }      
      case hash("--sort_samples"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          sort_samples = true;
        }else{
          sort_samples = false;     
        }               
        break;     
      }        
      case hash("--schedule_local_batch_first"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          schedule_local_batch_first = true;
        }else{
          schedule_local_batch_first = false;     
        }               
        break;     
      }       
      case hash("--sync_by_cond"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          sync_by_cond = true;
        }else{
          sync_by_cond = false;     
        }               
        break;     
      }      
      case hash("--include_accuracy"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          loadrun_args.include_accuracy = true;
        }else{
          loadrun_args.include_accuracy = false;     
        }               
        break;     
      }      
      case hash("--msg_buffer_size"):
        msg_buffer_size = std::stoi(argv[++index])*sizeof(Msg_Size); 
        break;    
      case hash("--instance_loadgen_makeup"):
        instance_loadgen_makeup = std::stoi(argv[++index]); 
        break;        
      case hash("--mlperf_config_file_name"):{
        std::string temp(argv[++index]);      
        mlperf_config_file_name = temp; 
        use_mlperf_config_file = true;
        break;         
      }
      case hash("--user_config_file_name"):{
        std::string temp(argv[++index]);      
        user_config_file_name = temp; 
        use_user_config_file = true;
        break;         
      }      
      case hash("--model_name"):{
        std::string temp(argv[++index]);                   
        model_name = temp;
        break;                 
      }
      default:        
        inferencer_argv.push_back(argv[index]);
        inferencer_argc++;
        break;
    }
  };

  loadrun_args.batch_size = loadrun_args.loadrun_queue_size;
  if(use_mlperf_config_file){
    mlperf_args.FromConfig(mlperf_config_file_name, model_name, scenario_name);
    std::cout << "\nmlperf config file " << mlperf_config_file_name << " used \n";         
  } else{
    std::cout << "\nno mlperf config file is available \n";            
  }
  if(use_user_config_file){
    mlperf_args.FromConfig(user_config_file_name, model_name, scenario_name);
    std::cout << "\nuser config file " << user_config_file_name << " used \n\n";         
  } else{
    std::cout << "\nno user config file is available \n\n";            
  }  
  std::cout << "schedule_local_batch_first is " << schedule_local_batch_first << "\n";

  if (!run_as_instance){
    // single process code for singlestream scenario
    sut_classifier = get_inferencer();
    if (!sut_classifier){
      std::cout << "Error to get inferencer\n";  
      return 1;
    } 

    if(schedule_local_batch_first){
      sut_classifier->initialize(inferencer_argc, &inferencer_argv[0], true);
    }else{
      sut_classifier->initialize(inferencer_argc, &inferencer_argv[0], false);
    }    
  
    mlperf::SystemUnderTest *sut = (mlperf::SystemUnderTest *)mlperf::c::ConstructSUT(0,"CSUT", 4, 
                                    IssueQuery, FlushQueries, ReportLatencyResults);    
    mlperf::QuerySampleLibrary *qsl = (mlperf::QuerySampleLibrary *)mlperf::c::ConstructQSL(0, "CSUT", 4, 
                                        total_samples, performance_samples, LoadSamplesToRam, 
                                        UnloadSamplesFromRam);

    std::cout << "mlperf performance_samples is " << performance_samples << "\n";	
    std::cout << "mlperf total_samples is " << total_samples << "\n";	
    std::cout << "mlperf min_query_count is " << mlperf_args.min_query_count << "\n";	
    std::cout << "mlperf min_duration_ms is " << mlperf_args.min_duration_ms << "\n";	
    std::cout << "mlperf offline_expected_qps is " << mlperf_args.offline_expected_qps << "\n";	 
    std::cout << "mlperf server_target_qps is " << mlperf_args.server_target_qps << "\n";  

    ben_timer.start();
    mlperf::c::StartTest(sut, qsl, mlperf_args);
    ben_timer.end();
#ifdef PROFILE     
    double ben_seconds = ben_timer.get_ms_duration()/1000;
      
    std::cout << "\nLoadgen time is " << ben_seconds << " seconds\n";   

    double hd_seconds = 0;
    double run_seconds = 0;    
    float top1 = 0;
    float top5 = 0;
    sut_classifier->getInfo(&hd_seconds, &run_seconds, &top1, &top5);
    std::cout << "hw time is " << hd_seconds << " seconds\n";  
    std::cout << "run time is " << run_seconds << " seconds\n";     
    std::cout << "accuracy is " << top1/inference_samples << " \n";     
    std::cout << "inference samples " << inference_samples << " images\n";     
#endif
    std::cout << "\nCompleted\n\n";

    return 0;
  }
  
  //multi instanses code
  std::cout << "server name is " << server_name <<"\n";
  number_instances = ins_names.size();

  struct shm_remove{
      shm_remove() { 
        shared_memory_object::remove(server_name.c_str()); 
        for (size_t index = 0; index < ins_names.size(); index++){
          shared_memory_object::remove((ins_names[index] + "_command").c_str()); 
          shared_memory_object::remove((ins_names[index] + "_response").c_str()); 
          shared_memory_object::remove((ins_names[index] + "_status").c_str());      
          shared_memory_object::remove((ins_names[index] + "_sync").c_str());    
        }
        named_mutex::remove(("mtx" + server_name).c_str());
        named_mutex::remove(("mtx_cmd" + server_name).c_str());
        shared_memory_object::remove("MySharedCond");
      }
      ~shm_remove(){ 
        shared_memory_object::remove(server_name.c_str()); 
        for (size_t index = 0; index < ins_names.size(); index++){
          shared_memory_object::remove((ins_names[index] + "_command").c_str()); 
          shared_memory_object::remove((ins_names[index] + "_response").c_str()); 
          shared_memory_object::remove((ins_names[index] + "_status").c_str());          
          shared_memory_object::remove((ins_names[index] + "_sync").c_str());           
        }
        named_mutex::remove(("mtx" + server_name).c_str());
        named_mutex::remove(("mtx_cmd" + server_name).c_str());
        shared_memory_object::remove("MySharedCond");
      }
  } remover;
  (void)remover;

  std::vector<shared_memory_object> ins_cmd_shms, ins_rsp_shms, ins_status_shms, ins_sync_shms;
  std::vector<mapped_region> ins_cmd_regions, ins_rsp_regions, ins_status_regions, ins_sync_regions;

  for (size_t index = 0; index < ins_names.size(); index++){
    std::cout << "server start shared_memory_object cmd " << ins_names[index] + "_command" << "\n";
    ins_cmd_shms.push_back(shared_memory_object(create_only, 
                                                (ins_names[index] + "_command").c_str(), 
                                                read_write));
    ins_cmd_shms[index].truncate(msg_buffer_size);
    ins_cmd_regions.push_back(mapped_region(ins_cmd_shms[index], read_write));    
    std::memset(ins_cmd_regions[index].get_address(), 0, ins_cmd_regions[index].get_size());
    pos_c_cmd_vector.push_back(static_cast<Msg_Size*>(ins_cmd_regions[index].get_address()));    

    std::cout << "server start shared_memory_object rsp " << ins_names[index] << "\n";
    ins_rsp_shms.push_back(shared_memory_object (create_only, 
                                                (ins_names[index] + "_response").c_str(), 
                                                read_write));
    ins_rsp_shms[index].truncate(msg_buffer_size);
    ins_rsp_regions.push_back(mapped_region(ins_rsp_shms[index], read_write));
    std::memset(ins_rsp_regions[index].get_address(), 0, ins_rsp_regions[index].get_size());    
    pos_c_rsp_vector.push_back(static_cast<Msg_Size*>(ins_rsp_regions[index].get_address()));    

    std::cout << "server start shared_memory_object status " << ins_names[index] << "\n";
    ins_status_shms.push_back(shared_memory_object (create_only, 
                                                   (ins_names[index] + "_status").c_str(), 
                                                   read_write));
    ins_status_shms[index].truncate(msg_buffer_size);
    ins_status_regions.push_back(mapped_region(ins_status_shms[index], read_write));
    std::memset(ins_status_regions[index].get_address(), 0, ins_status_regions[index].get_size());    
    pos_c_status_vector.push_back(static_cast<Msg_Size*>(ins_status_regions[index].get_address()));    

    std::cout << "server start shared_memory_object sync " << ins_names[index] << "\n";
    ins_sync_shms.push_back(shared_memory_object (create_only, 
                                                   (ins_names[index] + "_sync").c_str(), 
                                                   read_write));
    ins_sync_shms[index].truncate(msg_buffer_size);
    ins_sync_regions.push_back(mapped_region(ins_sync_shms[index], read_write));
    std::memset(ins_sync_regions[index].get_address(), 0, ins_sync_regions[index].get_size());    
    pos_c_sync_vector.push_back(static_cast<Msg_Size*>(ins_sync_regions[index].get_address()));

    std::cout << "server start client cmd buffer at " << pos_c_cmd_vector[index] << "\n";
    std::cout << "server start client rsp buffer at " << pos_c_rsp_vector[index] << "\n";
    std::cout << "server start client status buffer at " << pos_c_status_vector[index] << "\n";    
    std::cout << "server start client sync buffer at " << pos_c_sync_vector[index] << "\n";
  }

  shared_memory_object server_shm (create_only, server_name.c_str(), read_write);
  server_shm.truncate(msg_buffer_size);
  mapped_region s_region(server_shm, read_write);  
  std::memset(s_region.get_address(), 0, s_region.get_size());
  pos_server_cmd = static_cast<Msg_Size*>(s_region.get_address());
  std::cout << "server start cmd buffer pointer " << pos_server_cmd << "\n" ;
  std::cout << "server start cmd buffer value " << *pos_server_cmd << "\n" ;

  shared_memory_object shm(create_only, "MySharedCond", read_write);
  shm.truncate(sizeof(trace_queue));
  mapped_region region(shm, read_write);
  void* addr = region.get_address();
  sync_cond = new (addr) trace_queue;
  sync_cond->instance_done_counter = ins_names.size();
  sync_cond->instance_loadsample_counter = ins_names.size();  
  std::cout << "server start sync cond pointer " << addr << "\n" ;
  std::cout << "server start sync instance inference counter is " << sync_cond->instance_done_counter << "\n" ;  
  std::cout << "server start sync instance loadsample counter is " << sync_cond->instance_loadsample_counter << "\n" ;   

  // when schedule_local_batch_first, response is checked in isseQuesy()
  if(!schedule_local_batch_first && !sync_by_cond){
    std::thread check_thread(check_response_by_thread, pos_c_rsp_vector);
    check_thread.detach();
  }

  mlperf::SystemUnderTest *sut = (mlperf::SystemUnderTest *)mlperf::c::ConstructSUT(0,"CSUT", 4, 
                                  IssueQuery, FlushQueries, ReportLatencyResults);    
  mlperf::QuerySampleLibrary *qsl = (mlperf::QuerySampleLibrary *)mlperf::c::ConstructQSL(0, "CSUT", 4, 
                                      total_samples, performance_samples, LoadSamplesToRam, 
                                      UnloadSamplesFromRam);

  std::cout << "\nmlperf performance_samples is " << performance_samples << "\n";	
  std::cout << "mlperf total_samples is " << total_samples << "\n";	
  std::cout << "mlperf override_min_query_count is " << mlperf_args.min_query_count << "\n";	
  std::cout << "mlperf override_min_duration_ms is " << mlperf_args.min_duration_ms << "\n";	
  std::cout << "mlperf offline_expected_qps is " << mlperf_args.offline_expected_qps << "\n";	 
  std::cout << "mlperf server_target_qps is " << mlperf_args.server_target_qps << "\n\n";              
  std::cout << "offline_single_response is " << offline_single_response << "\n";  
  std::cout << "loadrun_sleep_after_each_response_check is " << loadrun_sleep_after_each_response_check << "\n";
  std::cout << "instance_loadgen_makeup is " << instance_loadgen_makeup << "\n";    
  std::cout << "schedule_local_batch_first is " << schedule_local_batch_first << "\n";    
  std::cout << "sync_by_cond is " << sync_by_cond << "\n";  
  std::cout << "sort_samples is " << sort_samples << "\n\n";  

  if (interactive){
    std::cout << "Wait for start command\n";     
    while(true){   
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));  
      if ((*pos_server_cmd) == CMD_START){
        std::cout << "\nLoad started\n"; 
        break;
      }    
    } 
  } else{
    named_mutex named_mtx{open_or_create, ("mtx_cmd" + server_name).c_str()};
    named_mtx.lock();      
    *(pos_server_cmd + CMD_SYNC_OFFSET) += 1;
    std::cout << "Loadrun ready\n";     
    named_mtx.unlock();
    while(true){   
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));  
      
      // check loadrun + netrun instances started
      if (*(pos_server_cmd + CMD_SYNC_OFFSET) == (ins_names.size() + 1)){  
        named_mtx.lock();          
        (*pos_server_cmd) = CMD_START;
        named_mtx.unlock();                
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));      
        std::cout << "\nLoad started\n"; 
        break;
      }
    }    
  }
  
  ben_timer.start();
  mlperf::c::StartTest(sut, qsl, mlperf_args);
  ben_timer.end();

  std::cout << "\nWait for stop command\n";  
  if (!interactive){
    if (sync_by_cond){
      // notify all instances start to work
      scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);   
      sync_cond->cond_instance_start.notify_all(); 
      lock_instance_start.unlock();       
    }
    
    // notify other check response thread and netrun instances exit        
    (*pos_server_cmd) = CMD_STOP;
  } else {
    // wait for interractive cmd stop is issued
    while((*pos_server_cmd) != CMD_STOP){
      std::this_thread::sleep_for (std::chrono::seconds(1));
    } 
  }

  // wait for netrun instances stop
  while(*(pos_server_cmd + CMD_SYNC_OFFSET) > 1){   
    std::this_thread::sleep_for (std::chrono::seconds(1));
  }  

  // when schedule_local_batch_first or sync by condition, check response thread is not started
  if(!schedule_local_batch_first && !sync_by_cond){
    // notify check response thread stop
    *(pos_server_cmd + CMD_CHECK_THREAD_OFFSET) = CMD_STOP;      
    while(*(pos_server_cmd + CMD_LOADRUN_MAIN_OFFSET) != CMD_STOP){
      std::this_thread::sleep_for (std::chrono::seconds(1));
    }
  }

  // exit after check response thread ends
  named_mutex named_mtx{open_or_create, ("mtx" + server_name).c_str()};
  named_mtx.lock();   
#ifdef PROFILE  
  double ben_seconds = ben_timer.get_ms_duration()/1000;
  double ben_qps = inference_samples / ben_seconds;
  std::cout << "\nloadrun sort time is " << sort_timer.get_ms_duration() << " miliseconds\n";     
  std::cout << "loadrun loadgen-QuerySamplesComplete time is " << query_complete_timer.get_ms_duration() << " milliseconds\n";;	  
  std::cout << "loadrun " << server_name << " issue query time is " 
            << issue_query_timer.get_ms_duration() << " milliseconds\n";   
  std::cout << "fps is " << ben_qps << " images/second\n";	
  std::cout << "loadgen time is " << ben_seconds << " seconds\n";   
  std::cout << "inferences samples " << inference_samples << " imgs\n";
#endif
  std::cout << "\nCompleted\n\n";
  named_mtx.unlock();  
  
  return 0;  
}

