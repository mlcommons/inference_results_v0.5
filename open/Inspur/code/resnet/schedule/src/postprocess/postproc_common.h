#ifndef __PostProcCommon_H__
#define __PostProcCommon_H__

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>

#include "postprocess_base.h"


namespace postprocess {
	template<typename T>
	class  PostProcCommon: public PostProcessBase<T>
	{
	public:
		PostProcCommon(T offset);
		~PostProcCommon() ;

		void AddResults(vector<T>) override {}

		std::vector<T> RunCommon(shared_ptr<BatchImageNet> pred_results) override;                          //for imagenet_vgg

		virtual std::vector<std::vector<std::vector<std::vector<T>>>>UploadResultsCoco() { 
			std::vector<std::vector<std::vector<std::vector<T>>>> emptyVec;
			return emptyVec;
		}

		virtual void AddResultsCoco(std::vector<std::vector<std::vector<T>>> ResultOneBatch) {  }

		void Reset() override;

		//void finalize(std::unordered_map<std::string, int> &final_result);

		std::map<std::string, size_t> UploadResults() override;
	private:
		int64_t offset_;
		size_t good_num;
		size_t total_num;

	};

}


#endif  