#include "../cnpy/cnpy.h"

#include "../common/utils.h"
#include "../common/image_op.h"
#include "schedule.h"
#include "inference_node.h"
#include "data_set.h"

using namespace std;


namespace schedule {

	template<typename T>
	Batch<T>::Batch() {
		m_p_data = nullptr;
		m_p_data_org = nullptr;
		m_size = 0;
		m_size_org = 0;
		m_p_buff = nullptr;
	};

	template<typename T>
	Batch<T>::~Batch() {
		if (m_p_data_org)
			delete[] m_p_data_org;
		if (m_p_buff)
			cudaFree(m_p_buff);
	}

	template class Batch<QuerySample>;
	template class Batch<ImageData>;
	template class Batch<MemoryData>;
	template class Batch<PredictResult<float>>;
	template class Batch<PredictResult<inference::ResultTensor*>>;

	void DataSet::LoadQuerySamples(const QuerySampleIndex* samples, size_t size) {
		m_image_list_inmemory.clear();

		for (size_t i = 0; i < size; i++) {
			float* ret = GetItem(samples[i]);
			shared_ptr<float> sp(ret, SampleDataDeleter);
			m_image_list_inmemory[samples[i]] = sp;
		}
		m_last_loaded = common::ScheduleClock::now();
	}

	void DataSet::UnloadQuerySamples(const QuerySampleIndex* samples, size_t size) {
		if (size > 0 and samples) {
			for (size_t i = 0; i < size; i++) {
				m_image_list_inmemory.erase(samples[i]);
			}
		}
		else {
			m_image_list_inmemory.clear();
		}
	}

	shared_ptr<float> DataSet::GetSampleInput(size_t index) {
		return m_image_list_inmemory[index];
	}

	vector<float> DataSet::GetSampleLable(size_t index) {
		mp::MLPerfSettings& settings = Schedule::GetSchedule()->get_m_settings();
		return settings.get_m_label_list()[index];
	}

	string DataSet::GetItemLoc(size_t index) {
		mp::MLPerfSettings& settings = Schedule::GetSchedule()->get_m_settings();
		string src = common::Utils::PathJoin(2, settings.get_m_cache_dir(), settings.get_m_image_list()[index]);
		return src;
	}

	float* DataSet::GetItem(size_t index) {
		mp::MLPerfSettings& settings = Schedule::GetSchedule()->get_m_settings();
		string dst = common::Utils::PathJoin(2, settings.get_m_cache_dir().c_str(), settings.get_m_image_list()[index].c_str());
		
		// Read npy
		cnpy::NpyArray arr = cnpy::npy_load(dst + ".npy");
		float* loaded_data = arr.data<float>();
		//float* data = new float[arr.num_bytes() / sizeof(float)];
		float* data;
		if (cudaMallocHost((void **) &data, arr.num_bytes()) != cudaSuccess ) {
			  std::cout << "cudaMallocHost Error" << std::endl;
		}
		memcpy(data, loaded_data, arr.num_bytes());
		return data;

		// Read NHWC jpeg
/*		cv::Mat mat = cv::imread(dst);
                cout << dst << endl;
		assert(!mat.empty());

		InferenceNode* node = dynamic_cast<InferenceNode*>(Schedule::GetSchedule()->get_m_graph().GetNodesByType("Inference")[0]);
		string data_fomat = node->GetDataFormat();
		size_t channels = mat.channels();
		size_t rows = mat.rows;
		size_t cols = mat.cols;
		size_t elements_num = channels * rows * cols;
		float* data = new float[elements_num];
		shared_ptr<float> sp(data, SampleDataDeleter);
		uchar* start = mat.ptr<unsigned char>(0);
		if (data_fomat == "NCHW") {
			common::hwc2chw<uchar, float>(channels, cols, rows, start, data);
		}
		else {
			for (size_t i = 0; i < elements_num; i++)
			{
					data[i] = static_cast<float>(start[i]);
			}
		}

		if(settings.get_m_profile_name()=="ssd-mobilenet-tf")
		{
			for (size_t i = 0; i < elements_num; i++) 
			{
				data[i] = data[i] * 2 / 255 - 1.0;
			}
		}
        cout << "data: " << data[0] << *(sp.get()) << endl;
		return sp;*/
	}
}
