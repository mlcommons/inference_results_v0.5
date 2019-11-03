//#include "postproc_argmax.h"
//#include "postproc_common.h"
//
//
//namespace postprocess{
//	template<typename T>
//	PostProcArgMax<T>::PostProcArgMax(T offset) :offset_(offset) {
//		good_num = 0;
//		total_num = 0;
//	}
//
//	template<typename T>
//	PostProcArgMax<T>:: ~PostProcArgMax() {}
//
//	template<typename T>
//	std::vector<T> PostProcArgMax<T>::RunArgMax(shared_ptr<BatchImageNet> pred_results)
//	{
//		std::vector<T> processed_result;
//		vector<T> result_vec = argmax_2d(pred_results);
//
//		for (size_t i = 0; i < result_vec.size(); i++)
//		{
//			auto data = pred_results->get_m_p_data()[i];
//			auto result = result_vec[i] + offset_;
//			processed_result.push_back(result);
//			if (static_cast<int64_t>(result) == static_cast<int64_t>(data.get_m_label()[0])) good_num++;
//		}
//		total_num += pred_results->size();
//		return processed_result;
//	}
//
//	template<typename T>
//	std::vector<T> PostProcArgMax<T>::argmax_2d(shared_ptr<BatchImageNet> pred_results, int axis)
//	{
//		int resultSize = pred_results->get_m_size();
//		std::vector<T> result_vec;
//		result_vec.reserve(resultSize);
//		int maxElem = 0;
//		int argmax_ = 0;
//		for (int i = 0; i < resultSize; i++)
//		{
//			result_vec[i] = (std::max_element(pred_results.get()[i].begin(), result_plane[i].end()) - (result_plane[i]).begin());
//		}
//		return result_vec;
//	}
//
//	template<typename T>
//	void PostProcArgMax<T>::Reset() {
//		good_num = 0;
//		total_num = 0;
//	}
//
//	//template<typename T>
//	//void PostProcArgMax<T>::finalize(std::unordered_map<std::string, int> &final_result)
//	//{
//	//	final_result["good"] = good_num;
//	//	final_result["total"] = total_num;
//	//}
//
//	template<typename T>
//	std::map<std::string, size_t> PostProcArgMax<T>::UploadResults() {
//		std::map<std::string, size_t> result_list;
//		result_list.emplace("good", good_num);
//		result_list.emplace("total", total_num);
//		return result_list;
//	}
//
//	template class PostProcArgMax<float>;
//
//}
