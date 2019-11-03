#ifndef __PostProcBase_H__
#define __PostProcBase_H__

#include <map>
#include <vector>

#include "../inference/inference.h"
#include "../schedule/data_set.h"

using namespace inference;
namespace postprocess {

	using PredictResultImageNet = schedule::PredictResult<float>;
	using BatchImageNet = schedule::Batch<PredictResultImageNet>;
	using PredictResultCoco = schedule::PredictResult<inference::ResultTensor*>;
	using BatchCoco = schedule::Batch<PredictResultCoco>;

	template <typename T>
	class PostProcessBase {
	public:
		virtual ~PostProcessBase() {
		}

		virtual void AddResults(vector<T>) {
		}

		virtual void AddResultsCoco(std::vector<std::vector<std::vector<T>>> ResultOneBatch) {
		}

		virtual std::map<std::string, size_t> UploadResults() {
			return std::map<std::string, size_t>(); 
		}

		virtual std::vector<std::vector<std::vector<std::vector<T>>>> UploadResultsCoco() {
			return std::vector<std::vector<std::vector<std::vector<T>>>>();
		}

		virtual std::vector<T> RunCommon(shared_ptr<BatchImageNet> pred_results) {
			return std::vector<T>();
		}

		//virtual std::vector<T> RunArgMax(shared_ptr<BatchImageNet> pred_results) {
		//	return std::vector<T>();
		//}

		virtual std::vector<std::vector<std::vector<T>>> RunCoco(shared_ptr<BatchCoco> pred_results) {
			return std::vector<std::vector<std::vector<T>>>();
		}

		virtual void Reset() {
		}
	};

}





#endif
