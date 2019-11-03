#include "inference.h"
#include <chrono>

using namespace std::chrono;


namespace inference
{

	extern int verbosity;

	shared_ptr<GpuEngine> GpuEngine::InitImgGpuEngines(string model_path,
		string data_path,
		string profile_name,
		unsigned int max_batchsize,
		string backend,
		unsigned int threads,
		size_t gpu_num)
	{   
		void** pBuffers;
		pBuffers = new void*[3] { nullptr, nullptr , nullptr };
		const char* pConnect[2];
		Dims connect_dim[2];
		DataType type[2];
		size_t input_size; 
		size_t output_size;

		//read engine cache to deserialize cuda engine and create context
//		char* pTREngine_cache;
//		int cache_length;
//		TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
//		tie(pTREngine_cache,cache_length) = read_engine(model_path);
//		IRuntime* runtime = createInferRuntime(trt_logger);
//		ICudaEngine* pTREngine = runtime->deserializeCudaEngine((const void*)pTREngine_cache,cache_length,nullptr);
                ICudaEngine* pTREngine = loadEngine(model_path, -1);
		if(pTREngine == nullptr)
		{
			cerr << "deserializeCudaEngine not worked" << endl;
		}

		IExecutionContext* pTRContext = pTREngine->createExecutionContext();
		// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
		// of these, but in this case we know that there is exactly one input and one output.
		assert(pTREngine->getNbBindings() == 2); 
/*		for(int i = 0; i < pTREngine->getNbBindings(); i++)
		{
			for(int j = 0;j < 4;j++)
				connect_dim[i].d[j] = 0;
			pConnect[i] = pTREngine->getBindingName(i);
			connect_dim[i] = pTREngine->getBindingDimensions(i);
			type =  pTREngine->getBindingDataType(i);
			if(i == 0)
			{
			    input_size = (size_t)max_batchsize * connect_dim[0].d[0] * connect_dim[0].d[1] * connect_dim[0].d[2] * sizeof(type);
				cout << "input size: " << input_size << endl;
			}
			if (i != 0)
			{
				output_size = (size_t)max_batchsize * connect_dim[1].d[0] * sizeof(type);
				CHECK(cudaMalloc(&pBuffers[i], output_size));
			}
			cout << "[bindings: " << i << "] " << pConnect[i] << ", max_batchsize:" << max_batchsize
				<< ", connect_dim[i].d[0]" << connect_dim[i].d[0] << ", connect_dim[i].d[1]" << connect_dim[i].d[1] << endl;
		}
		for(int j = 0;j < 4;j++)
		{
			 connect_dim[0].d[j] = 0;
			 connect_dim[1].d[j] = 0;
		}*/
		connect_dim[0] = pTREngine->getBindingDimensions(0);
		connect_dim[1] = pTREngine->getBindingDimensions(1);
		type[0] = pTREngine->getBindingDataType(0);
		type[1] = pTREngine->getBindingDataType(1);
		input_size = (size_t)max_batchsize * connect_dim[0].d[0] * connect_dim[0].d[1] * connect_dim[0].d[2] * sizeof(type[0]);
		output_size = (size_t)max_batchsize * connect_dim[1].d[0] * sizeof(type[1]);
		CHECK(cudaMalloc(&pBuffers[1], output_size));
		vector<ResultTensor> rts;
		rts.reserve(max_batchsize);

		cout << "data: input buffer: " << pBuffers[0] << ", output buffer1: " << pBuffers[1] << ", outputbuffer2: " << pBuffers[2]
			<< ", pTRContext: " << pTRContext << ", input_size: " << input_size << endl;

		float* h_output = new float[output_size];

		cudaStream_t stream;                                           //create local stream
		CHECK(cudaStreamCreate(&stream));

		shared_ptr<GpuEngine> pGpuEngine = make_shared<GpuEngine>(pTREngine,
			pTRContext,
			(float*)NULL,
			pBuffers[0],
			h_output,
			pBuffers[1],
			stream,
			(size_t)max_batchsize,
			input_size,
			output_size,
			(int)threads,
			pBuffers,
			false,
			profile_name,
			gpu_num,
			rts);

		return pGpuEngine;
	}

	void GpuEngine::doInference(shared_ptr<Batch<MemoryData>> input)
	{
		int h_output_length =  h_output_size / sizeof(float) / max_batchsize;
		int input_batch_size = input->get_m_size();
		assert(input_batch_size <= max_batchsize);
		int real_output_bytes = h_output_length * sizeof(float) * input_batch_size;
		int64_t result = 0;
		bindings[0] = input->get_m_p_buff();
//		IExecutionContext* context = engine->createExecutionContext();
//		cudaStream_t stream;
//		CHECK(cudaStreamCreate(&stream));
//		context->execute(input_batch_size, bindings);
    		context->enqueue(input_batch_size, bindings, stream, nullptr);
    		
//		CHECK(cudaMemcpy(h_output, d_output, real_output_bytes, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpyAsync(h_output, d_output, real_output_bytes, cudaMemcpyDeviceToHost, stream));
		CHECK(cudaStreamSynchronize(stream));
//		CHECK(cudaStreamDestroy(stream));
		result_tensor.clear();
		for(int i = 0; i < input_batch_size; i++)
		{
		   result = (int64_t)distance(&h_output[i * h_output_length], max_element(&h_output[i * h_output_length], &h_output[(i + 1) * h_output_length]));
		   result_tensor.push_back({result, nullptr, nullptr, nullptr});
		}
	}

}
