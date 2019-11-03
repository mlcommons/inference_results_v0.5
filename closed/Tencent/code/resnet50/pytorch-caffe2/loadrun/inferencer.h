#ifndef LOADRUN_INFERENCER_H_
#define LOADRUN_INFERENCER_H_

#include <memory>

namespace loadrun {

class Inferencer {
  public:
    Inferencer(){};
    virtual ~Inferencer(){};
    
    // pass the fw related parameters to backends for initialization, warmpup
    // use_index == true, initialize will defer samples loading to LoadSamplesToRam()
    virtual void initialize(int argc, char **argv, bool use_index) = 0;

    // send the index of sample to backend
    virtual void prepare_batch(const int index) = 0;
    
    // run fw inference
    // For random samples in ram, 
    //     set random = true, call prepare_batch(each sample index) by batch size times before run()
    //     set iteration = 0    
    // For continuous samples in ram, 
    //     set random = false, call prepare_batch(batch index) one time before run()
    //     set iteration = batch index
    virtual void run(int iteration, const bool random) = 0;

    // get accuracy data, not used at performance mode
    // For random samples in ram, 
    //     set random = true
    //     set iteration = 0    
    // For continuous samples in ram, 
    //     set random = false
    //     set iteration = batch index
    virtual void accuracy(int iteration, bool random) = 0;   
    
    // get inference results, the order is as same as the sequence of calling prepare_batch()
    virtual std::vector<int> get_labels(int iteration, const bool random) = 0;
    
    // load samples to ram by the sequence specified by samples
    // samples - a pointer to the 1st sample index
    // sample_size - the number of samples
    virtual void load_sample(size_t* samples, size_t sample_size) = 0;

    // get fw related info, optional
    // hd_seconds and run_seconds returns fw internal timer value, return 0 if not available
    // top1 and top5 returns top accuracy, not used at performance mode
    virtual void getInfo(double* hd_seconds, float* top1, float* top5)  = 0;
    virtual void getInfo(double* run_seconds, double* hd_seconds, float* top1, float* top5)  = 0;   
};

} // name space loadrun

std::unique_ptr<loadrun::Inferencer> get_inferencer();

#endif  // LOADRUN_INFERENCER_H_