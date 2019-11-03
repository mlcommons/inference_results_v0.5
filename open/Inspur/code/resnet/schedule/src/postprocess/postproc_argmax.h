//#ifndef __PostProcArgMax_H__
//#define __PostProcArgMax_H__
//
//#include <vector>
//#include <string>
//#include <algorithm>
//#include <iostream>
//#include <map>
//#include "postprocess_base.h"
//
//namespace postprocess {
//	template<typename T>
//	class  PostProcArgMax: public PostProcessBase<T>
//	{
//	public:
//		PostProcArgMax(T offset);
//		~PostProcArgMax() ;
//
//		void AddResults(vector<T>) override {}
//
//		std::vector<T> RunArgMax(shared_ptr<BatchImageNet> pred_results) override; //for imagenet_mobilenet
//
//		void Reset() override;
//
//		//void finalize(std::unordered_map<std::string, int> &final_result);
//
//		std::map<std::string, size_t> UploadResults() override;
//
//		//virtual std::vector<std::vector<std::vector<T>>> RunCoco(std::vector<CoCoResult<T>> pred_results, std::vector<size_t> ids, std::vector<std::vector<T>> expected) {
//		//	std::vector<std::vector<std::vector<T>>> emptyVec;
//		//	return emptyVec;
//		//}
//
//		virtual void AddResultsCoco(std::vector<std::vector<std::vector<T>>> ResultOneBatch) {  }
//
//		//virtual std::vector<T> RunCommon(std::vector<T> pred_results, std::vector<size_t> ids, std::vector<T> expected) {
//		//	std::vector<T> emptyVec;
//		//	return emptyVec;
//		//};                        
//
//		virtual std::vector<std::vector<std::vector<std::vector<T>>>>UploadResultsCoco() {
//			std::vector<std::vector<std::vector<std::vector<T>>>> emptyVec;
//			return emptyVec;
//		}
//
//	private:
//		std::vector<T> argmax_2d(shared_ptr<BatchImageNet> pred_results, int axis = 1);
//
//		int64_t offset_;
//		int good_num;
//		int total_num;
//
//	};
//
//
//}
//
//
//
//#endif