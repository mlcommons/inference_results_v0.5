#ifndef __INTERFACE_H
#define __INTERFACE_H
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <new>
#include <cassert>
#include <fstream>  
#include <iomanip>
#include <sstream>
#include <future>
#include <thread>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cstdlib>
#include <exception>
#include <type_traits>
#include <stdbool.h>

//#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "NvInfer.h"
#include "./ssd_Calibration/common.h"
#include "../schedule/data_set.h"


namespace inference
{
	using namespace std;
	using namespace nvinfer1;
	using namespace schedule;
	// using namespace nvcaffeparser1;
	typedef struct result_struct
	{
		vector<int> detection_nums;
		vector<vector<vector<int>>> detection_boxes;
		vector<float> detection_scores;
		vector<string> detection_classes;
	}result_struct;
	struct ResultTensor
	{
		int64_t num;
		float* boxes;
		float* scores;
		float* detected_classes;
	};
	
	struct InferDeleter
	{
		template <typename T>
		void operator()(T* obj) const
		{
			if (obj)
			{
				obj->destroy();
			}
		}
	};

	template <typename T>
	struct destroyer
	{
		void operator()(T* t) { t->destroy(); }
	};

	class GpuEngine
	{
		public:
			GpuEngine();
			GpuEngine(ICudaEngine* engine,
				IExecutionContext* context,
				float* h_input,     // CPU input memory
				void* d_input,      // GPU input memory
				float* h_output,    // CPU output memory
				void* d_output,     // GPU output memory
				cudaStream_t stream,
				size_t max_batchsize,
				size_t h_input_size,
				size_t h_output_size,
				int thread_num,
				void* bindings[3],
				bool exist,
				string profile,
				int gpu_num,
				vector<ResultTensor> result_tensor);
			~GpuEngine(){
				cudaStreamDestroy(stream);
				delete h_output;
				}
		   static vector<shared_ptr<GpuEngine>> InitGpuEngines(string model_path,
			   string data_path,
			   string profile_name,
			   unsigned int max_batchsize,
			   string backend,
			   unsigned int threads);

		   static shared_ptr<GpuEngine> InitImgGpuEngines(string model_path,
			   string data_path,
			   string profile_name,
			   unsigned int max_batchsize,
			   string backend,
			   unsigned int threads,
			   size_t gpu_num);

			static shared_ptr<GpuEngine> InitSSDGpuEngines(string model_path,
				string data_path,
				string profile_name,
				uint max_batchsize,
				string profile,
				uint threads,
				size_t gpu_num);

			void doInference(shared_ptr<Batch<MemoryData>> input);
			void doSSDInference(shared_ptr<Batch<MemoryData>> input);
			vector<ResultTensor>* Predict(shared_ptr<Batch<MemoryData>> input);
			vector<size_t> doTestInference(vector<shared_ptr<cv::Mat>> input,int max_batchsize,float* loaded_data);
			static vector<float*> test(void* input,int max_batchsize,void *param);
			size_t GetInputSize() { return h_input_size; }
			size_t GetOutputSize() { return h_output_size; }

		private:
			ICudaEngine* engine;
			IExecutionContext* context;
			float* h_input;
			void* d_input;
			float* h_output;
			void* d_output;
	      		cudaStream_t stream;
			int max_batchsize;
			size_t h_input_size;
			size_t h_output_size;
			int thread;
			void** bindings;
			bool exist = false;
			string profile;
			int gpu_num;
			vector<ResultTensor> result_tensor;
	};

	class TRT_Logger : public nvinfer1::ILogger {
		nvinfer1::ILogger::Severity _verbosity;
		ostream* _ostream;

		public:
			TRT_Logger(Severity verbosity=Severity::kWARNING,
					   ostream& ostream=cout)
			  : _verbosity(verbosity), _ostream(&ostream) {}

			void log(Severity severity, const char* msg) override {
			  if( severity <= _verbosity ) {
				time_t rawtime = time(0);
				char buf[256];
				strftime(&buf[0], 256,
						 "%Y-%m-%d %H:%M:%S",
						 gmtime(&rawtime));
				const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" :
									  severity == Severity::kERROR          ? "  ERROR" :
									  severity == Severity::kWARNING        ? "WARNING" :
									  severity == Severity::kINFO           ? "   INFO" :
									  "UNKNOWN");
				(*_ostream) << "[" << buf << " " << sevstr << "] "
							<< msg
							<< endl;
			  }
			}
		};
		tuple<char*,int> read_engine(string model_path);
                ICudaEngine* loadEngine(const std::string& engine, int DLACore);

}
#endif
