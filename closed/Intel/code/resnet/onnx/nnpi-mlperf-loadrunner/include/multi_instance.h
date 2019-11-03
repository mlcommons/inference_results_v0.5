#ifndef LOADRUN_MULTI_INSTANCE_H_
#define LOADRUN_MULTI_INSTANCE_H_

#include "query_sample.h"

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

namespace loadrun {

struct LoadrunSettings {
  // reserve different queue size and batch size for scheduling experiment
  Msg_Size loadrun_queue_size = 1;
  Msg_Size batch_size = 1;

  bool is_offline = false;
  Msg_Size offline_samples = std::numeric_limits<Msg_Size>::max();
};

} // name space loadrun

#endif  // LOADRUN_MULTI_INSTANCE_H_