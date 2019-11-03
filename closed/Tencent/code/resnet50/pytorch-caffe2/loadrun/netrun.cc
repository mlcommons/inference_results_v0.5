#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>

// multi instances support
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "multi_instance.h"
#include <boost/interprocess/detail/config_begin.hpp>
#include <cstring>

// loadgen integration
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

// backend integration
#include "inferencer.h"
 
double hd_seconds = 0;
double run_seconds = 0;
float top1 = 0;
float top5 = 0;

// multi instances support
loadrun::LoadrunSettings loadrun_args;
bool interactive = false;
bool schedule_local_batch_first = false;
mlperf::TestSettings mlperf_args;
std::string ins_name("");
std::string server_name("");
Msg_Size * pos_c_cmd = nullptr;
Msg_Size * pos_c_rsp = nullptr;
Msg_Size *pos_server_cmd = nullptr;
Msg_Size batch_size = 1;
Msg_Size * pos_c_status = nullptr;
std::vector<Msg_Size> responses_id_vector;
bool sync_by_cond = false;

local_timer inf_timer;

extern std::unique_ptr<loadrun::Inferencer> get_inferencer();

uint64_t constexpr mix(char m, uint64_t s) {
  return ((s<<7) + ~(s>>3)) + ~m;
}
 
uint64_t constexpr hash(const char * m) {
  return (*m) ? mix(*m,hash(m+1)) : 0;
}

int main(int argc, char **argv) { 
  using namespace boost::interprocess;
  bool stop_all = false;
  bool start_run = false;  
  // sleep for a while after sending a batch of reponse
  bool netrun_sleep_after_each_response_send = true;
    
  // move inferencer related args to new array for backend initialization  
  std::vector<char *> inferencer_argv;
  int inferencer_argc = 0;
  for(int index = 0; index < argc; index++ ){     
    switch(hash(argv[index])){                    
      case hash("--instance"):{
        ins_name = std::string(argv[++index]);
        }
        break;  
      case hash("--server"):{
        server_name = std::string(argv[++index]);        
        }
        break;
      case hash("--stop_all"):{
        std::string str(argv[++index]);      
        if (str.compare("true") == 0){
          stop_all = true;
        }               
        break;     
      }
      case hash("--start_run"):{
        std::string str(argv[++index]);      
        if (str.compare("true") == 0){
          start_run = true;
        }               
        break;  
      }             
      case hash("--batch_size"):
        // batch size is also used by inferencer
        inferencer_argv.push_back(argv[index]);
        batch_size = std::stoi(argv[++index]); 
        inferencer_argv.push_back(argv[index]);
        inferencer_argc += 2;        
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
      case hash("--netrun_sleep_after_each_response_send"):{
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          netrun_sleep_after_each_response_send = true;
        }else{
          netrun_sleep_after_each_response_send = false;     
        }               
        break;                 
      }           
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
      case hash("--sync_by_cond"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          sync_by_cond = true;
        }else{
          sync_by_cond = false;     
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
      case hash("--include_accuracy"):{        
        std::string temp(argv[++index]);      
        if (temp.compare("true") == 0){
          loadrun_args.include_accuracy = true;
        }else{
          loadrun_args.include_accuracy = false;     
        }               
        break;     
      }           
      default:  
        inferencer_argv.push_back(argv[index]);
        inferencer_argc++;
        break;
    }
  };

  std::cout << "client open server cmd buffer " << server_name + "_command" << "\n";
  shared_memory_object server_shm (open_only, server_name.c_str(), read_write);
  mapped_region region_server(server_shm, read_write);
  pos_server_cmd = static_cast<Msg_Size*>(region_server.get_address());

  if(start_run){  
    // issue start to loadrun and all netrun instances
    std::cout << "start\n";      
    (*pos_server_cmd) = CMD_START;
    return 0;
  }  

  if(stop_all){  
    // issue stop to loadrun and all netrun instances    
    std::cout << "Stop\n";    
    (*pos_server_cmd) = CMD_STOP;
    return 0;
  } 

  std::cout << "client start shared_memory_object cmd " << ins_name + "_command" << "\n";
  shared_memory_object c_cmd_shm (open_only, (ins_name + "_command").c_str(), read_write);
  mapped_region region_c_cmd(c_cmd_shm, read_write);
  pos_c_cmd = static_cast<Msg_Size*>(region_c_cmd.get_address()); 
  std::cout << "client " << ins_name << " cmd buffer address" << pos_c_cmd << "\n";  

  shared_memory_object c_rsp_shm (open_only, (ins_name + "_response").c_str(), read_write);
  mapped_region region_c_rsp(c_rsp_shm, read_write);
  pos_c_rsp = static_cast<Msg_Size*>(region_c_rsp.get_address()); 
  std::cout << "client " << ins_name << " response buffer address" << pos_c_rsp << "\n"; 

  shared_memory_object c_status_shm (open_only, (ins_name + "_status").c_str(), read_write);
  mapped_region region_c_status(c_status_shm, read_write);
  pos_c_status = static_cast<Msg_Size*>(region_c_status.get_address());
  std::cout << "client " << ins_name << " status buffer address " << pos_c_status << "\n";  

  shared_memory_object shm(open_only, "MySharedCond", read_write);
  mapped_region region(shm, read_write);
  void* addr = region.get_address();
  trace_queue* sync_cond = static_cast<trace_queue*>(addr);
  std::cout << "client " << ins_name << " sync cond address " << sync_cond << "\n";  

  auto sut_classifier = get_inferencer();
  if (!sut_classifier){
    std::cout << "Error to get inferencer\n";  
    return 1;
  }
  if(schedule_local_batch_first){
    sut_classifier->initialize(inferencer_argc, &inferencer_argv[0], true);
    std::cout << "Netrun initialize done with sample index\n";
  }else{
    sut_classifier->initialize(inferencer_argc, &inferencer_argv[0], false);
    std::cout << "Netrun initialize done without sample index\n";
  }
  std::cout << "Netrun initialize done\n";

  // notify server initilization and warmup is done
  named_mutex named_mtx{open_or_create, ("mtx_cmd" + server_name).c_str()};
  named_mtx.lock();      
  *(pos_server_cmd + CMD_SYNC_OFFSET) += 1;
  std::cout << *(pos_server_cmd + CMD_SYNC_OFFSET) -1 << " Netrun ready\n";     
  named_mtx.unlock();

  std::cout << "netrun_sleep_after_each_response_send is " << netrun_sleep_after_each_response_send << "\n";  
  std::cout << "schedule_local_batch_first is " << schedule_local_batch_first << "\n";   
  std::cout << "sync_by_cond is " << sync_by_cond << "\n";  
  
  if (!sync_by_cond){
    std::cout << "Wait for start command\n";     
    while(true){
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
      if ((*pos_server_cmd) == CMD_START){
        break;
      }    
    } 
  }else{  
    // block self and wait for next server completes query dispatch
    std::cout << "Wait for start condition\n";     
    scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);        
    sync_cond->cond_instance_start.wait(lock_instance_start); 
    lock_instance_start.unlock();
  }

  Msg_Size total_number = 0;  
  Msg_Size accumulated_number = 0;
  while (true){
    // offline with continuous samples in ram
    if (schedule_local_batch_first){
      // offline mode only, for loading coninuous samples to ram 
      // for this version of loadgen, LoadSamplesToRam() is called once before 
      // IssueQueries() in performance mode
      // for Submission mode, LoadSamplesToRam() will be called twice, one for
      // Accuracy, another for performance
      if((*pos_c_cmd) == CMD_LOAD_SAMPLES){
        pos_c_cmd++;
        size_t sample_size = *pos_c_cmd;
        pos_c_cmd++;
        sut_classifier->load_sample(pos_c_cmd, sample_size);
        pos_c_cmd+=sample_size;
        
        if(sync_by_cond){
          // notify server loadsample is done
          scoped_lock<interprocess_mutex> lock_loadsample(sync_cond->loadsample_mutex);
          sync_cond->instance_loadsample_counter--;   
          if (sync_cond->instance_loadsample_counter == 0)
            sync_cond->cond_loadsample_done.notify_one();
          lock_loadsample.unlock();

          // block self and wait for server completes query dispatch
          scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);        
          sync_cond->cond_instance_start.wait(lock_instance_start);  
          lock_instance_start.unlock();  
        }else{
          *(pos_c_status + STATUS_LOAD_SAMPLES_OFFSET) = STATUS_LOAD_SAMPLES_DONE;
        }
      }
      
      // offline mode only, for coninuous samples in ram
      // Buffer cmd format
      //    CMD_SCHEDULE_LOCAL_BATCH_FIRST
      //    length
      //    start batch index      
      if((*pos_c_cmd) == CMD_SCHEDULE_LOCAL_BATCH_FIRST){
        pos_c_cmd++;

        // local batch first, there's batch index in cmd buffer 
        Msg_Size length = *pos_c_cmd;
        pos_c_cmd++;
        Msg_Size start_batch = *pos_c_cmd;
        pos_c_cmd++;  
        
        for (size_t index=0; index < length; index++){
          Msg_Size index_batch = start_batch + index;
          sut_classifier->prepare_batch(index_batch);   

          inf_timer.start();                       
          sut_classifier->run(index_batch, false); 
          inf_timer.end();          

          // fixit decide the code after same code at accuracy/performance ls clarified      
          if(loadrun_args.include_accuracy){
            std::vector<int> results = sut_classifier->get_labels(index_batch, false);  
            for (size_t i_respose=0; i_respose < results.size(); i_respose++){
              (*pos_c_rsp) = results[i_respose]; 
              pos_c_rsp++;        
            }        
          }
          total_number += batch_size;
        }

        (*pos_c_status) = STATUS_DONE;   
        
        if(sync_by_cond){
          // all samples are inferenced
          scoped_lock<interprocess_mutex> lock(sync_cond->instance_done_mutex);
          sync_cond->instance_done_counter--;
          if (sync_cond->instance_done_counter == 0){
            sync_cond->cond_instance_done.notify_one();                   
          }
          lock.unlock();

          // block self and wait for next server IssueQuery()
          scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);        
          sync_cond->cond_instance_start.wait(lock_instance_start); 
          lock_instance_start.unlock();
        }
      }    
    }
    
    // offline with incontinuous samples in ram
    if(!schedule_local_batch_first && sync_by_cond){
      // round robin by sample, there're response id + sample index in cmd buffer
      while ((*pos_c_cmd) != 0 && (*(pos_c_cmd + 1)) != 0){
        responses_id_vector.push_back(*pos_c_cmd); 
        pos_c_cmd++;

        // -1 for +1 in loadrun IssueQuery() to handle img index 0      
        sut_classifier->prepare_batch((*pos_c_cmd) - 1);      
  
        pos_c_cmd++;
        accumulated_number++;
        total_number++; 
          
        // get a batch
        if (accumulated_number == batch_size){     
          inf_timer.start();          
          sut_classifier->run(0, true);              
          inf_timer.end();

          // send back response
          if(loadrun_args.include_accuracy){                    
            std::vector<int> results = sut_classifier->get_labels(0, true);
            for (size_t i_respose=0; i_respose < responses_id_vector.size(); i_respose++){
              (*pos_c_rsp) = responses_id_vector[i_respose]; 
              pos_c_rsp++;
              (*pos_c_rsp) = results[i_respose]; 
              pos_c_rsp++;        
            }        
          }else{
            for (auto id:responses_id_vector) {          
              (*pos_c_rsp) = id; 
              pos_c_rsp++;              
            }
          }
          accumulated_number = 0;          
          responses_id_vector.clear();          
        }
      }

      // handle remain batch for indivisable batch size
      if (accumulated_number < batch_size){         
        inf_timer.start();        
        sut_classifier->run(0, true);              
        inf_timer.end();

        // send back response
        if(loadrun_args.include_accuracy){                    
          std::vector<int> results = sut_classifier->get_labels(0, true);
          for (size_t i_respose=0; i_respose < responses_id_vector.size(); i_respose++){
            (*pos_c_rsp) = responses_id_vector[i_respose]; 
            pos_c_rsp++;
            (*pos_c_rsp) = results[i_respose]; 
            pos_c_rsp++;        
          }        
        }else{
          for (auto id:responses_id_vector) {          
            (*pos_c_rsp) = id; 
            pos_c_rsp++;              
          }
        }
        accumulated_number = 0;          
        responses_id_vector.clear();
      }

      // all samples are inferenced
      scoped_lock<interprocess_mutex> lock(sync_cond->instance_done_mutex);
      sync_cond->instance_done_counter--;
      if (sync_cond->instance_done_counter == 0){
        sync_cond->cond_instance_done.notify_one();                   
      }
      lock.unlock();

      // block self and wait for next server IssueQuery()
      scoped_lock<interprocess_mutex> lock_instance_start(sync_cond->instance_start_mutex);        
      sync_cond->cond_instance_start.wait(lock_instance_start);
      lock_instance_start.unlock();     
    }

    // Server and SingleStream
    if (!schedule_local_batch_first && !sync_by_cond){
      // round robin by sample, there're response id + sample index in cmd buffer
      while ((*pos_c_cmd) != 0 && (*(pos_c_cmd + 1)) != 0){
        responses_id_vector.push_back(*pos_c_cmd); 
        pos_c_cmd++;

        // -1 for +1 in loadrun IssueQuery() to handle img index 0      
        sut_classifier->prepare_batch((*pos_c_cmd) - 1);      
  
        pos_c_cmd++;
        accumulated_number++;
        total_number++;       
    
        // get a full batch or loadgen ends
        if (accumulated_number == batch_size){     
          inf_timer.start();
          sut_classifier->run(0, true);                        
          inf_timer.end();          

          // send back response
          if(loadrun_args.include_accuracy){                    
            std::vector<int> results = sut_classifier->get_labels(0, true);
            for (size_t i_respose=0; i_respose < responses_id_vector.size(); i_respose++){
              (*pos_c_rsp) = responses_id_vector[i_respose]; 
              pos_c_rsp++;
              // handle inference result 0
              (*pos_c_rsp) = results[i_respose] + 1; 
              pos_c_rsp++;        
            }        
          }else{
            for (auto id:responses_id_vector) {          
              (*pos_c_rsp) = id; 
              pos_c_rsp++;              
            }
          }
          accumulated_number = 0;          
          responses_id_vector.clear();          

        }        
      }// end while
      
      // handle remaining samples 
      if ((*(pos_server_cmd + CMD_TEST_END_OFFSET) == CMD_TEST_END) && (accumulated_number > 0)){
        inf_timer.start();        
        sut_classifier->run(0, true);                        
        inf_timer.end();

        // send back response
        if(loadrun_args.include_accuracy){                    
          std::vector<int> results = sut_classifier->get_labels(0, true);
          for (size_t i_respose=0; i_respose < responses_id_vector.size(); i_respose++){
            (*pos_c_rsp) = responses_id_vector[i_respose]; 
            pos_c_rsp++;
            // handle inference result 0
            (*pos_c_rsp) = results[i_respose] + 1; 
            pos_c_rsp++;        
          }        
        }else{
          for (auto id:responses_id_vector) {          
            (*pos_c_rsp) = id; 
            pos_c_rsp++;              
          }
        }
        accumulated_number = 0;          
        responses_id_vector.clear();          
      }
    }// end of server and singlestream

    if ((*pos_server_cmd) == CMD_STOP){
      named_mutex named_mtx{open_or_create, ("mtx" + server_name).c_str()};
      named_mtx.lock();
      sut_classifier->getInfo(&run_seconds, &hd_seconds, &top1, &top5);
#ifdef PROFILE       
      std::cout << "\ninstance " << ins_name << " inference time is " << inf_timer.get_ms_duration() << " milliseconds\n";        
      std::cout << "instance " << ins_name << " processed " << total_number << " samples\n";    
      std::cout << "accuracy is " << top1/total_number << " \n";        
      std::cout << "instance " << ins_name << " hw time is " << hd_seconds << " seconds\n";  
      std::cout << "s instance " << ins_name << " runtime time is " << run_seconds << " seconds\n";  
#endif
      std::cout << "\ninstance " << ins_name << " done\n";           
      *(pos_server_cmd + CMD_SYNC_OFFSET) -= 1;     
      named_mtx.unlock();  
      return 0;
    }
    
    if (netrun_sleep_after_each_response_send){
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));          
    }
  }    
}