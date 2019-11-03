# pragma once
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../cnpy/cnpy.h"
#include "../common/macro.h"
#include "../common/types.h"
#include "settings/mlperf_settings.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace std;


namespace schedule {

	class ImageData {
	public:
		CREATE_SIMPLE_ATTR_SET_GET(m_p_sample, QuerySample*)
		CREATE_SIMPLE_ATTR_SET_GET(m_p_mat, shared_ptr<float>)
		CREATE_SIMPLE_ATTR_SET_GET(m_label, vector<float>)
		CREATE_SIMPLE_ATTR_SET_GET(m_tp, common::ScheduleClock::time_point)
	};

	class MemoryData {
	public:
		CREATE_SIMPLE_ATTR_SET_GET(m_p_sample, QuerySample*)
		CREATE_SIMPLE_ATTR_SET_GET(m_label, vector<float>)
		CREATE_SIMPLE_ATTR_SET_GET(m_tp, common::ScheduleClock::time_point)
	};

	template<typename T>
	class PredictResult {
	public:
		CREATE_SIMPLE_ATTR_SET_GET(m_p_sample, QuerySample*)
		CREATE_SIMPLE_ATTR_SET_GET(m_result, T)
		CREATE_SIMPLE_ATTR_SET_GET(m_label, vector<float>)
		CREATE_SIMPLE_ATTR_SET_GET(m_tp, common::ScheduleClock::time_point)
	};

	auto SampleDataDeleter = [](float* p) {
		//delete[] p;
		cudaFreeHost(p);
	};

	auto QuerySampleResponseDeleter = [](QuerySampleResponse* p) {
		delete[] p;
	};

	template<typename T>
	class Batch {
		CREATE_SIMPLE_ATTR_SET_GET(m_p_data, T*)
		CREATE_SIMPLE_ATTR_SET_GET(m_p_data_org, T*)
		CREATE_SIMPLE_ATTR_SET_GET(m_size, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_size_org, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_tp, common::ScheduleClock::time_point)
		CREATE_SIMPLE_ATTR_SET_GET(m_p_buff, void*)

	public:
		Batch();
		~Batch();
	};

	class DataSet {
		using IndexDataMap = map<size_t, shared_ptr<float>>;

		CREATE_SIMPLE_ATTR_SET_GET(m_arrival, void*)
		CREATE_SIMPLE_ATTR_SET_GET(m_image_list, vector<string>)
		CREATE_SIMPLE_ATTR_SET_GET(m_label_list, vector<float>)
		CREATE_SIMPLE_ATTR_SET_GET(m_image_list_inmemory, IndexDataMap)
		CREATE_SIMPLE_ATTR_SET_GET(m_last_loaded, common::ScheduleClock::time_point)

	public:
		DataSet() {
			m_arrival = nullptr;
			m_last_loaded = common::ScheduleClock::now();
		};
		~DataSet() {};

		size_t GetItemCount() { return this->m_image_list.size(); }
		void LoadQuerySamples(const QuerySampleIndex* samples, size_t size);
		void UnloadQuerySamples(const QuerySampleIndex* samples, size_t size);
		shared_ptr<float> GetSampleInput(size_t index);
		vector<float> GetSampleLable(size_t index);
		virtual string GetItemLoc(size_t index);

	protected:
		virtual float* GetItem(size_t index);
	};

}
