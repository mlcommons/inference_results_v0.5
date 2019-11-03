#include "postproc_coco.h"
#include "../common/logging.h"


namespace postprocess {
	using Logger = common::Logger;

	template<typename T>
	PostProcessCoco<T>::PostProcessCoco() {
		good = 0;
		total = 0;
		results.clear();
		use_inv_map_ = false;
	}

	template<typename T>
	PostProcessCoco<T>::~PostProcessCoco() {}


	bool FindExpected(vector<float> labels, float target) {
		for (size_t i = 0; i < labels.size(); i++) {
			if (static_cast<int64_t>(labels[i]) == static_cast<int64_t>(target)) 
				return true;
		}
		return false;
	}

	template<typename T>
	std::vector<std::vector<std::vector<T>>> PostProcessCoco<T>::RunCoco(shared_ptr<BatchCoco> pred_results)
	{
		std::vector<std::vector<std::vector<T>>> postprocCoCoResult;
		
		auto batch_size = pred_results->get_m_size();
		
		for (size_t idx = 0; idx < batch_size; idx++) {
			std::vector<std::vector<T>> oneimage;
			postprocCoCoResult.push_back(oneimage);
			auto p_data = pred_results->get_m_p_data()[idx];
			auto data_result = p_data.get_m_result();
			auto detect_num = data_result->num;
			
			for (int i = 0; i < p_data.get_m_label().size(); i++) {
				Logger::Log(__FILE__, __FUNCTION__, "expected labels:", p_data.get_m_label()[i]);
			}
			Logger::Log(__FILE__, __FUNCTION__, "detections num in one pic:", detect_num);

			for (int64_t detection = 0; detection < detect_num; detection++) {
				std::vector<T> onedetection;
				postprocCoCoResult[idx].push_back(onedetection);
				auto detect_class = *(data_result->detected_classes + detection);
				if (FindExpected(p_data.get_m_label(), detect_class)) good++;
				postprocCoCoResult[idx][detection].push_back(p_data.get_m_p_sample()->index);
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 0));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 1));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 2));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 3));
				postprocCoCoResult[idx][detection].push_back(*(data_result->scores + detection));
				postprocCoCoResult[idx][detection].push_back(detect_class);
				total++;
				Logger::Log(__FILE__, __FUNCTION__, "detection class:", detect_class);
				Logger::Log(__FILE__, __FUNCTION__, "detection index:", p_data.get_m_p_sample()->index);
				Logger::Log(__FILE__, __FUNCTION__, "dectection scores:", *(data_result->scores + detection));
				Logger::Log(__FILE__, __FUNCTION__, "box_0:", *(data_result->boxes + 4 * detection + 0));
				Logger::Log(__FILE__, __FUNCTION__, "box_1:", *(data_result->boxes + 4 * detection + 1));
				Logger::Log(__FILE__, __FUNCTION__, "box_2:", *(data_result->boxes + 4 * detection + 2));
				Logger::Log(__FILE__, __FUNCTION__, "box_3:", *(data_result->boxes + 4 * detection + 3));
			}
		}
		return postprocCoCoResult;
	}


	template<typename T>
	std::vector<std::vector<std::vector<std::vector<T>>>> PostProcessCoco<T>::UploadResultsCoco() {
		return results;
	}

	template<typename T>
	std::map<std::string, size_t> PostProcessCoco<T>::UploadResults() {
		std::map<std::string, size_t> result_list;
		result_list.emplace("good", good);
		result_list.emplace("total", total);
		return result_list;
	}

	template<typename T>
	void PostProcessCoco<T>::AddResultsCoco(std::vector<std::vector<std::vector<T>>> postprocresultOneBatch)
	{
		results.push_back(postprocresultOneBatch);
	}

	template<typename T>
	void PostProcessCoco<T>::Reset() {
		good = 0;
		total = 0;
		results.clear();
	}


	template<typename T>
	PostProcessCocoPt<T>::PostProcessCocoPt(bool use_inv_map, T score_threshold) :
		PostProcessCoco<T>(), score_threshold_(score_threshold)
	{
		PostProcessCoco<T>::use_inv_map_ = use_inv_map;
	}

	template<typename T>
	PostProcessCocoPt<T>::~PostProcessCocoPt() {}

	template<typename T>
	std::vector<std::vector<std::vector<T>>>  PostProcessCocoPt<T>::RunCoco(shared_ptr<BatchCoco> pred_results) {
		std::vector<std::vector<std::vector<T>>> postprocCoCoResult;
		auto batch_size = pred_results->get_m_size();
		for (size_t idx = 0; idx < batch_size; idx++)
		{
			std::vector<std::vector<T>> oneImage;
			postprocCoCoResult.push_back(oneImage);
			auto p_data = pred_results->get_m_p_data()[idx];
			auto data_result = p_data.get_m_result();
			auto detect_num = data_result->num;
			for (int64_t detection = 0; detection < detect_num; detection++)
			{
				std::vector<T> onedetection;
				postprocCoCoResult[idx].push_back(onedetection);
				auto detect_class = *(data_result->detected_classes + detection);
				if (*(data_result->scores + detection) < score_threshold_) continue;
				if (FindExpected(p_data.get_m_label(), detect_class)) PostProcessCoco<T>::good++;
				postprocCoCoResult[idx][detection].push_back(p_data.get_m_p_sample()->index);
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 1));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 0));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 3));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 2));
				postprocCoCoResult[idx][detection].push_back(*(data_result->scores + detection));
				postprocCoCoResult[idx][detection].push_back(detect_class);
				PostProcessCoco<T>::total++;
			}
		}

		return postprocCoCoResult;
	}

	template<typename T>
	PostProcessCocoOnnx<T>::PostProcessCocoOnnx() :PostProcessCoco<T>() { };

	template<typename T>
	PostProcessCocoOnnx<T>::~PostProcessCocoOnnx() {};

	template<typename T>
	std::vector<std::vector<std::vector<T>>>  PostProcessCocoOnnx<T>::RunCoco(shared_ptr<BatchCoco> pred_results) {
		std::vector<std::vector<std::vector<T>>> postprocCoCoResult;
		auto batch_size = pred_results->get_m_size();
		for (size_t idx = 0; idx < batch_size; idx++)
		{
			std::vector<std::vector<T>> oneImage;
			postprocCoCoResult.push_back(oneImage);
			auto p_data = pred_results->get_m_p_data()[idx];
			auto data_result = p_data.get_m_result();
			auto detect_num = data_result->num;
			for (int64_t detection = 0; detection < detect_num; detection++)
			{
				std::vector<T> onedetection;
				postprocCoCoResult[idx].push_back(onedetection);
				auto detect_class = *(data_result->detected_classes + detection);
				if (*(data_result->scores + detection) < 0.5) continue;
				if (FindExpected(p_data.get_m_label(), detect_class)) PostProcessCoco<T>::good++;
				postprocCoCoResult[idx][detection].push_back(p_data.get_m_p_sample()->index);
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 1));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 0));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 3));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 2));
				postprocCoCoResult[idx][detection].push_back(*(data_result->scores + detection));
				postprocCoCoResult[idx][detection].push_back(detect_class);
				PostProcessCoco<T>::total++;
			}
		}

		return postprocCoCoResult;
	}

	template<typename T>
	PostProcessCocoTf<T> ::PostProcessCocoTf() :PostProcessCoco<T>() {
		PostProcessCoco<T>::use_inv_map_ = true;
	}

	template<typename T>
	PostProcessCocoTf<T> ::~PostProcessCocoTf() {}

	template<typename T>
	std::vector<std::vector<std::vector<T>>>  PostProcessCocoTf<T>::RunCoco(shared_ptr<BatchCoco> pred_results) {
		std::vector<std::vector<std::vector<T>>> postprocCoCoResult;
		auto batch_size = pred_results->get_m_size();
		for (size_t idx = 0; idx < batch_size; idx++)
		{
			std::vector<std::vector<T>> oneImage;
			postprocCoCoResult.push_back(oneImage);
			auto p_data = pred_results->get_m_p_data()[idx];
			auto data_result = p_data.get_m_result();
			auto detect_num = data_result->num;
			for (int64_t detection = 0; detection < detect_num; detection++)
			{
				std::vector<T> onedetection;
				postprocCoCoResult[idx].push_back(onedetection);
				auto detect_class = *(data_result->detected_classes + detection);
				if (*(data_result->scores + detection) < 0.05) continue;
				if (FindExpected(p_data.get_m_label(), detect_class)) PostProcessCoco<T>::good++;
				postprocCoCoResult[idx][detection].push_back(p_data.get_m_p_sample()->index);
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 0));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 1));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 2));
				postprocCoCoResult[idx][detection].push_back(*(data_result->boxes + 4 * detection + 3));
				postprocCoCoResult[idx][detection].push_back(*(data_result->scores + detection));
				postprocCoCoResult[idx][detection].push_back(detect_class);
				PostProcessCoco<T>::total++;
			}
		}

		return postprocCoCoResult;
	}

	template class PostProcessCoco<float>;
	template class PostProcessCocoPt<float>;
	template class PostProcessCocoOnnx<float>;
	template class PostProcessCocoTf<float>;
}

