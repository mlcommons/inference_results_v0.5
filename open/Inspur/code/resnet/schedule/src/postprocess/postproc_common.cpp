#include "../common/logging.h"
#include "postproc_common.h"


namespace postprocess{

	using Logger = common::Logger;

	template<typename T>
	PostProcCommon<T>::PostProcCommon(T offset) :offset_(offset) {
		good_num = 0;
		total_num = 0;
	}

	template<typename T>
	PostProcCommon<T>:: ~PostProcCommon() {}

	template<typename T>
	std::vector<T> PostProcCommon<T>::RunCommon(shared_ptr<BatchImageNet> pred_results) {
		std::vector<T> processed_result;
		auto size = pred_results->get_m_size();

		for (size_t i = 0; i < size; i++)
		{
			auto data = pred_results->get_m_p_data()[i];
			auto result = data.get_m_result() + offset_;
			processed_result.push_back(result);
			if (static_cast<int64_t>(result) == static_cast<int64_t>(data.get_m_label()[0])) good_num++;
			//std::cout << "predict:" << result << " label:" << data.get_m_label()[0] << std::endl;

		}
		total_num += size;
		return processed_result;
	}

	template<typename T>
	void PostProcCommon<T>::Reset() {
		good_num = 0;
		total_num = 0;
	}

	template<typename T>
	std::map<std::string, size_t> PostProcCommon<T>::UploadResults() {
		std::map<std::string, size_t> result_list;
		result_list.emplace("good", good_num);
		result_list.emplace("total", total_num);

		return result_list;
	}

	template class PostProcCommon<float>;

}

