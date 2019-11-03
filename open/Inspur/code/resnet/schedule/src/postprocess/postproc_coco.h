#ifndef __PostProcCoco_H__
#define __PostProcCoco_H__


#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <fstream>
#include "postprocess_base.h"

//struct PostProcResult
//{
//	float ids;
//	float* box;
//	float score;
//	float detected_class;
//};


namespace postprocess {

	//template<type>
	//void cocoevalCallBack(std::string file_name, std::vector<T>);

	template<typename T>
	class DataSource
	{
	public:
		DataSource() {}
		~DataSource() {}
		std::vector<T> image_ids;
		std::vector<std::vector<T>> image_sizes;
		std::vector<std::vector<T>> image_loc;

		std::string annotation_file; //json file address
		std::vector<T> get_item_loc(int ID) {
			return image_loc[ID];
		}
	private:

	};
	template class DataSource<float>;

	


	template<typename T>
	class PostProcessCoco:public PostProcessBase<T>
	{
	public:
		PostProcessCoco();
		virtual ~PostProcessCoco();

		std::vector<std::vector<std::vector<T>>> RunCoco(shared_ptr<BatchCoco> pred_results) override;

		void AddResultsCoco(std::vector<std::vector<std::vector<T>>> postprocresultOneBatch) override;

		void Reset() override;

		std::vector<std::vector<std::vector<std::vector<T>>>>UploadResultsCoco() override;

		std::map<std::string, size_t> UploadResults() override;

		void AddResults(vector<T>) override {}

		//virtual std::vector<T> RunArgMax(std::vector<std::vector<T>> pred_result, std::vector<size_t> ids, std::vector<T> expected) {
		//	std::vector<T> emptyVec;
		//	return emptyVec;
		//}

		//virtual std::vector<T> RunCommon(std::vector<T> pred_results, std::vector<size_t> ids, std::vector<T> expected) {
		//	std::vector<T> emptyVec;
		//	return emptyVec;
		//};



	protected:

		int good;
		int total;
		std::vector<std::vector<std::vector<std::vector<T>>>> results;
		bool use_inv_map_;

	};

	template<typename T>
	class PostProcessCocoPt : public PostProcessCoco<T> {
	public:
		PostProcessCocoPt(bool use_inv_map, T score_threshold);
		virtual ~PostProcessCocoPt();
		std::vector<std::vector<std::vector<T>>> RunCoco(shared_ptr<BatchCoco> pred_results) override;

	protected:
		T score_threshold_;
	};


	template<typename T>
	class PostProcessCocoOnnx : public PostProcessCoco<T> {
	public:
		PostProcessCocoOnnx();
		virtual ~PostProcessCocoOnnx();
		std::vector<std::vector<std::vector<T>>> RunCoco(shared_ptr<BatchCoco> pred_results) override;
	};


	template<typename T>
	class PostProcessCocoTf : public PostProcessCoco<T> {
	public:
		PostProcessCocoTf();
		virtual ~PostProcessCocoTf();
		std::vector<std::vector<std::vector<T>>> RunCoco(shared_ptr<BatchCoco> pred_results) override;
	};


}



#endif