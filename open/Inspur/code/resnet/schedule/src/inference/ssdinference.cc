#include <map>
#include <chrono>
#include <float.h>
#include <string.h>
#include <valarray>
#include <exception>
#include <algorithm>

#include "./ssd_Calibration/BatchStreamPPM.h"
#include "./ssd_Calibration/common.h"
#include "./ssd_Calibration/argsParser.h"
#include "./ssd_Calibration/EntropyCalibrator.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "inference.h"
#include "ssdinference.h"


namespace inference
{

	using namespace nvuffparser;

	extern int verbosity;
	static constexpr int CAL_BATCH_SIZE = 10;
	static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;
	const char* INPUT_BLOB_NAME = "Input";

	string locateFile(const string& input)
	{
		vector<string> dirs{"./",
							"data/ssd/",
							"data/ssd/VOC2007/",
							"data/ssd/VOC2007/PPMImages/",
							"data/samples/ssd/",
							"data/int8_samples/ssd/",
							"data/samples/ssd/VOC2007/",
							"data/samples/ssd/VOC2007/PPMImages/"};
		return locateFile(input);
	}

	class FlattenConcat : public IPluginV2
	{
	public:
		FlattenConcat(int concatAxis, bool ignoreBatch)
			: mIgnoreBatch(ignoreBatch)
			, mConcatAxisID(concatAxis)
		{
			assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
		}

		//clone constructor
		FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, int* inputConcatAxis)
			: mIgnoreBatch(ignoreBatch)
			, mConcatAxisID(concatAxis)
			, mOutputConcatAxis(outputConcatAxis)
			, mNumInputs(numInputs)
		{
			CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
			for (int i = 0; i < mNumInputs; ++i)
				mInputConcatAxis[i] = inputConcatAxis[i];
		}

		FlattenConcat(const void* data, size_t length)
		{
			const char *d = reinterpret_cast<const char*>(data), *a = d;
			mIgnoreBatch = read<bool>(d);
			mConcatAxisID = read<int>(d);
			assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
			mOutputConcatAxis = read<int>(d);
			mNumInputs = read<int>(d);
			CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
			CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

			for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

			mCHW = read<nvinfer1::DimsCHW>(d);

			for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

			assert(d == a + length);
		}

		~FlattenConcat()
		{
			if (mInputConcatAxis)
				CHECK(cudaFreeHost(mInputConcatAxis));
			if (mCopySize)
				CHECK(cudaFreeHost(mCopySize));
		}

		int getNbOutputs() const override { return 1; }

		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
		{
			assert(nbInputDims >= 1);
			assert(index == 0);
			mNumInputs = nbInputDims;
			CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
			mOutputConcatAxis = 0;

			for (int i = 0; i < nbInputDims; ++i)
			{
				int flattenInput = 0;
				assert(inputs[i].nbDims == 3);
				if (mConcatAxisID != 1)
					assert(inputs[i].d[0] == inputs[0].d[0]);
				if (mConcatAxisID != 2)
					assert(inputs[i].d[1] == inputs[0].d[1]);
				if (mConcatAxisID != 3)
					assert(inputs[i].d[2] == inputs[0].d[2]);
				flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
				mInputConcatAxis[i] = flattenInput;
				mOutputConcatAxis += mInputConcatAxis[i];
			}

			return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
						   mConcatAxisID == 2 ? mOutputConcatAxis : 1,
						   mConcatAxisID == 3 ? mOutputConcatAxis : 1);
		}

		int initialize() override
		{
			CHECK(cublasCreate(&mCublas));
			return 0;
		}

		void terminate() override
		{
			CHECK(cublasDestroy(mCublas));
		}

		size_t getWorkspaceSize(int) const override { return 0; }

		int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
		{
			int numConcats = 1;
			assert(mConcatAxisID != 0);
			numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());
			cublasSetStream(mCublas, stream);

			if (!mIgnoreBatch)
				numConcats *= batchSize;

			float* output = reinterpret_cast<float*>(outputs[0]);
			int offset = 0;
			for (int i = 0; i < mNumInputs; ++i)
			{
				const float* input = reinterpret_cast<const float*>(inputs[i]);
				float* inputTemp;
				CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

				CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

				for (int n = 0; n < numConcats; ++n)
				{
					CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
									  inputTemp + n * mInputConcatAxis[i], 1,
									  output + (n * mOutputConcatAxis + offset), 1));
				}
				CHECK(cudaFree(inputTemp));
				offset += mInputConcatAxis[i];
			}

			return 0;
		}

		size_t getSerializationSize() const override
		{
			return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
		}

		void serialize(void* buffer) const override
		{
			char *d = reinterpret_cast<char*>(buffer), *a = d;
			write(d, mIgnoreBatch);
			write(d, mConcatAxisID);
			write(d, mOutputConcatAxis);
			write(d, mNumInputs);
			for (int i = 0; i < mNumInputs; ++i)
			{
				write(d, mInputConcatAxis[i]);
			}
			write(d, mCHW);
			for (int i = 0; i < mNumInputs; ++i)
			{
				write(d, mCopySize[i]);
			}
			assert(d == a + getSerializationSize());
		}

		void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
		{
			assert(nbOutputs == 1);
			mCHW = inputs[0];
			assert(inputs[0].nbDims == 3);
			CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
			for (int i = 0; i < nbInputs; ++i)
			{
				mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
			}
		}

		bool supportsFormat(DataType type, PluginFormat format) const override
		{
			return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
		}

		const char* getPluginType() const override { return "FlattenConcat_TRT"; }

		const char* getPluginVersion() const override { return "1"; }

		void destroy() override { delete this; }

		IPluginV2* clone() const override
		{
			return new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis);
		}

		void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

		const char* getPluginNamespace() const override { return mNamespace.c_str(); }

	private:
		template <typename T>
		void write(char*& buffer, const T& val) const
		{
			*reinterpret_cast<T*>(buffer) = val;
			buffer += sizeof(T);
		}

		template <typename T>
		T read(const char*& buffer)
		{
			T val = *reinterpret_cast<const T*>(buffer);
			buffer += sizeof(T);
			return val;
		}

		size_t* mCopySize = nullptr;
		bool mIgnoreBatch{false};
		int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
		int* mInputConcatAxis = nullptr;
		nvinfer1::Dims mCHW;
		cublasHandle_t mCublas;
		string mNamespace;
	};

	namespace
	{
		const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
		const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
	} // namespace

	class FlattenConcatPluginCreator : public IPluginCreator
	{
	public:
		FlattenConcatPluginCreator()
		{
			mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
			mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));
			mFC.nbFields = mPluginAttributes.size();
			mFC.fields = mPluginAttributes.data();
		}

		~FlattenConcatPluginCreator() {}

		const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

		const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

		const PluginFieldCollection* getFieldNames() override { return &mFC; }

		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
		{
			const PluginField* fields = fc->fields;
			for (int i = 0; i < fc->nbFields; ++i)
			{
				const char* attrName = fields[i].name;
				if (!strcmp(attrName, "axis"))
				{
					assert(fields[i].type == PluginFieldType::kINT32);
					mConcatAxisID = *(static_cast<const int*>(fields[i].data));
				}
				if (!strcmp(attrName, "ignoreBatch"))
				{
					assert(fields[i].type == PluginFieldType::kINT32);
					mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
				}
			}
			return new FlattenConcat(mConcatAxisID, mIgnoreBatch);
		}

		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
		{
			//This object will be deleted when the network is destroyed, which will
			//call Concat::destroy()
			return new FlattenConcat(serialData, serialLength);
		}

		void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

		const char* getPluginNamespace() const override { return mNamespace.c_str(); }

	private:
		static PluginFieldCollection mFC;
		bool mIgnoreBatch{false};
		int mConcatAxisID;
		static vector<PluginField> mPluginAttributes;
		string mNamespace = "";
	};

	PluginFieldCollection FlattenConcatPluginCreator::mFC{};
	std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

	REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);

	vector<size_t> GpuEngine::doTestInference(vector<shared_ptr<cv::Mat>> input,int max_batchsize,float* loaded_data)
	{
		vector<size_t> output;
		int result = 0;
		int h_output_length =  h_output_size/sizeof(float);
		CHECK(cudaMemcpyAsync(d_input, input[0].get()->data, h_input_size, cudaMemcpyHostToDevice));
		context->enqueue(max_batchsize,bindings, stream, nullptr);
		CHECK(cudaMemcpyAsync(h_output, d_output, h_output_size, cudaMemcpyDeviceToHost));
		CHECK(cudaStreamSynchronize(stream));
		for(int i = 0;i < max_batchsize;i++)
		{
		   result = (size_t)distance(&h_output[i*h_output_length+1], max_element(&h_output[i*h_output_length], &h_output[i*h_output_length] + h_output_length));
		   output.push_back(result);
		}
		return output;
	}

	vector<float*> GpuEngine::test(void* input,int max_batchsize,void *param)
	{
		vector<float*> ll;
		ll.push_back((float*)&max_batchsize);
		return ll;
	}

	shared_ptr<GpuEngine> GpuEngine::InitSSDGpuEngines(std::string model_path,
		std::string data_path,
		std::string profile_name,
		uint max_batchsize,
		std::string backend,
		uint threads,
		size_t gpu_num)
	{   
		void** pBuffers;
		pBuffers = new void* [3]{ nullptr, nullptr , nullptr };
		const char* pConnect[3];
		Dims connect_dim[3];
		DataType type;
		size_t input_size; 
		int size_temp;
		size_t output_size;
		char* pTREngine_cache;
		int cache_length;

		initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

		TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
		tie(pTREngine_cache,cache_length) = read_engine(model_path);
		IRuntime* runtime = createInferRuntime(trt_logger);
		ICudaEngine* pTREngine = runtime->deserializeCudaEngine((const void*)pTREngine_cache,cache_length,nullptr);
		if(pTREngine == nullptr)
		{
			cerr << "deserializeCudaEngine not worked" << endl;
		}

		IExecutionContext* pTRContext = pTREngine->createExecutionContext();
			// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
			// of these, but in this case we know that there is exactly one input and one output.
		for(int i = 0; i < pTREngine->getNbBindings(); i++)
		{
			for(int j = 0;j < 4;j++)
			   connect_dim[i].d[j] = 0;
			pConnect[i] = pTREngine->getBindingName(i);
			connect_dim[i] = pTREngine->getBindingDimensions(i);
			type =  pTREngine->getBindingDataType(i);
			if(i == 0)
			{
			    size_temp = max_batchsize * connect_dim[0].d[0];
			    input_size = (size_t)size_temp * connect_dim[0].d[1] * connect_dim[0].d[2] * sizeof(type);
			    cout << "input size: " << input_size << endl;
//			    CHECK(cudaMalloc(&pBuffers[i],input_size));
			}
			if (i != 0)
			{
				size_temp = max_batchsize * connect_dim[0].d[0];
				output_size = (size_t)size_temp * connect_dim[1].d[1] * connect_dim[1].d[2]  * sizeof(type);
				CHECK(cudaMalloc(&pBuffers[i],output_size));
			}
			cout << "[bindings: " << i << "] " << pConnect[i] << ", max_batchsize:" << max_batchsize
				<< ", connect_dim[i].d[0]" << connect_dim[i].d[0] << ", connect_dim[i].d[1]" << connect_dim[i].d[1]
				<< ", connect_dim[i].d[2]" << connect_dim[i].d[2] << endl;
		}

		cout << "data: input buffer: " << pBuffers[0] << ", output buffer1: " << pBuffers[1] << ", outputbuffer2: " << pBuffers[2]
			<< ", pTRContext: " << pTRContext << ", input_size: " << input_size << endl;

		float* h_output = new float[output_size];

		vector<ResultTensor> rts;
		rts.reserve(max_batchsize);

		float* detection_scores = new float[sizeof(float) * 91 * max_batchsize];
		float* detection_classes = new float[sizeof(float) * 91 * max_batchsize];
		float* boxes = new float[sizeof(float) * 91 * max_batchsize * 4];

		for (uint p = 0; p < max_batchsize; p++)
		{
			rts.push_back({0, boxes + p * 91 * 4, detection_scores + p * 91, detection_classes + p * 91});
		}

		//generated the engineer vector
		cudaStream_t stream;
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

	void populateClassLabels(std::string (&CLASSES)[91])
	{
		auto fileName = locateFile("ssd_coco_labels.txt");
		ifstream labelFile(fileName);
		string line;
		int id = 0;
		while (getline(labelFile, line))
		{
			CLASSES[id++] = line;
		}
		return;
	}

	void GpuEngine::doSSDInference(shared_ptr<Batch<MemoryData>> input)
	{
		DetectionOutputParameters detectionOutputParam {true, false, 0, 91, 100, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};
		vector<float> detectionOut(max_batchsize * detectionOutputParam.keepTopK * 7);
		vector<int> keepCount(max_batchsize);
		float* pDetectionOut = (float*)&detectionOut[0]; 
		int* pKeepCount = (int*)&keepCount[0];
		const float visualizeThreshold = 0.5;
		int input_batch_size = input->get_m_size();
		bindings[0] = input->get_m_p_buff();
		context->execute(input_batch_size, bindings);

//		CHECK(cudaMemcpy(pDetectionOut, bindings[1], input_batch_size * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost));
//		CHECK(cudaMemcpy(pKeepCount, bindings[2], input_batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpyAsync(pDetectionOut, bindings[1], input_batch_size * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaMemcpyAsync(pKeepCount, bindings[2], input_batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaStreamSynchronize(stream));

		for (int p = 0; p < input_batch_size; p++)
		{
			int64_t numDetections_onePic = 0;
			for (int i = 0; i < keepCount[p]; ++i)
			{
				float* det = &detectionOut[0] + (p * detectionOutputParam.keepTopK + i) * 7;
			    if (det[2] < visualizeThreshold)
				  continue;
				assert((int) det[1] < 91);

				result_tensor[p].detected_classes[numDetections_onePic] = det[1];
				result_tensor[p].scores[numDetections_onePic] = det[2];
				result_tensor[p].boxes[4 * numDetections_onePic + 0] = det[3];
				result_tensor[p].boxes[4 * numDetections_onePic + 1] = det[4];
				result_tensor[p].boxes[4 * numDetections_onePic + 2] = det[5];
				result_tensor[p].boxes[4 * numDetections_onePic + 3] = det[6];
				numDetections_onePic++;
			}
			result_tensor[p].num = numDetections_onePic;
		}
	}

}
