#include <chrono>
#include "inference.h"

using namespace std;
using namespace nvinfer1;
using namespace std::chrono;


namespace inference
{

	int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
	
	ICudaEngine* loadEngine(const std::string& engine, int DLACore)
	{
		std::ifstream engineFile(engine, std::ios::binary);

		if (!engineFile)
		{
			cout << "Error opening engine file: " << engine << endl;
			return nullptr;
		}

		engineFile.seekg(0, engineFile.end);
		long int fsize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);

		std::vector<char> engineData(fsize);
		engineFile.read(engineData.data(), fsize);

		if (!engineFile)
		{
			cout << "Error loading engine file: " << engine << endl;
			return nullptr;
		}

		unique_ptr<IRuntime, destroyer<IRuntime>> runtime{ createInferRuntime(gLogger.getTRTLogger()) };
		if (DLACore != -1)
		{
			runtime->setDLACore(DLACore);
		}

		return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
	}
	
	GpuEngine::GpuEngine() {
	}

	GpuEngine::GpuEngine(ICudaEngine* engine,
		IExecutionContext* context,
		float* h_input,
		void* d_input,
		float* h_output,
		void* d_output,
		cudaStream_t stream,
		size_t max_batchsize,
		size_t h_input_size,
		size_t h_output_size,
		int thread_num,
		void* bindings[3],
		bool exist,
		string profile,
		int gpu_num,
		vector<ResultTensor> result_tensor)
			: engine(engine),
			context(context),
			h_input(h_input),
			d_input(d_input),
			h_output(h_output),
			d_output(d_output),
			stream(stream),
			max_batchsize(max_batchsize),
			h_input_size(h_input_size),
			h_output_size(h_output_size),
			thread(thread_num),
			bindings(bindings),
			exist(exist),
			profile(profile),
			gpu_num(gpu_num),
			result_tensor(result_tensor){
			}

	tuple<char*,int> read_engine(string model_path)
	{
		stringstream TREngine_file;
		ifstream TREngineCache(model_path.c_str());
		TREngine_file << TREngineCache.rdbuf(); 
		TREngineCache.close(); 

		// ifstream engine_file(model.c_str());
		if (!TREngine_file) {
			cerr << "Failed to open output file for reading: "<< model_path << endl;
		}

       
		TREngine_file.seekg(0, ios::end);
		int TREngine_file_length=TREngine_file.tellg();
		TREngine_file.seekg(0, ios::beg);
		cout << "TREngine_file_length: " << TREngine_file_length << endl;
		char* pTREngine_cache = (char*)malloc(sizeof(char)*TREngine_file_length);
		TREngine_file.read((char*)pTREngine_cache, TREngine_file_length);
		// engine_file.close();

		// free(pTREngine_cache);
		return tuple<char*,int>(pTREngine_cache,TREngine_file_length);
	}

	vector<shared_ptr<GpuEngine>> GpuEngine::InitGpuEngines(string model_path,
		string data_path,
		string profile_name,
		unsigned int max_batchsize,
		string backend,
		unsigned int threads)
	{
		int gpu_num = 0;
		CHECK(cudaGetDeviceCount(&gpu_num));
		vector<shared_ptr<GpuEngine>> engines;
		for(unsigned int i = 0;i < threads;i++)
		{
			cudaSetDevice(int(i % gpu_num));
			if(profile_name=="resnet50-tf"||profile_name=="mobilenet-tf")
				engines.push_back(GpuEngine::InitImgGpuEngines(model_path, data_path, profile_name, max_batchsize, backend, threads, gpu_num));
//			else if(profile_name=="ssd-mobilenet-tf"||profile_name=="ssd-resnet34-tf")
//				engines.push_back(GpuEngine::InitSSDGpuEngines(model_path, data_path, profile_name, max_batchsize, backend, threads, gpu_num));
		}
		return engines;
	}

	vector<ResultTensor>* GpuEngine::Predict(shared_ptr<Batch<MemoryData>> input)
	{
		chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

		if(profile=="resnet50-tf"||profile=="mobilenet-tf")
			doInference(input);
//		else if(profile=="ssd-mobilenet-tf"||profile=="ssd-resnet34-tf")
//			doSSDInference(input);

		chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(finish - start);
		cout<< "inference duration is: " <<  duration.count() / 1000.0 << "(ms)" << endl;
		return &result_tensor;
	}

}
