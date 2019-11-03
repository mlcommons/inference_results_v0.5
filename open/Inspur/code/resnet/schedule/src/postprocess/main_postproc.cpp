//#include "postproc_common.h"
//#include "postproc_coco.h"
//#include "postproc_argmax.h"
//#include "postprocess_base.h"
//
//int main()
//{
//	// 造一组batch数据predict_result_3d，测试PostProcCommon和PostProcArgMax
//
//	std::vector<int> expected(10);
//	for (size_t i = 0; i < 10; i++) 
//	{
//		expected[i] = 1;
//	}
//	//int one_result;
//	std::vector<std::vector<std::vector<int>>> predict_result_3d;
//	for (size_t i = 0; i < 2; i++)
//	{
//		std::vector<std::vector<int>> _2d_vec;
//		predict_result_3d.push_back(_2d_vec);
//		//predict_result_3d[i].resize(3);
//		for (size_t j = 0; j < 3; j++)
//		{
//			std::vector <int> _1d_vec;
//			predict_result_3d[i].push_back(_1d_vec);
//			for (int k = 0; k < 3; k++)
//			{
//				auto one_item = (k % 2 == 0) ? 0 : 1;
//				predict_result_3d[i][j].push_back(one_item);
//				std::cout << predict_result_3d[i][j][k] << " , ";
//			}
//			std::cout << std::endl;
//		}
//	}
//	
//	std::map<std::string, size_t> final_result;
//	std::map<std::string, size_t> final_result1;
//	std::map<std::string, size_t> final_result2;
//	std::map<std::string, size_t> final_result3;
//
//	//test PostProcCommon and PostProcArgMax
//	int offset = 0;
//	postprocess::PostProcessBase<int>* postprocessObj;
//	postprocessObj = new postprocess::PostProcCommon<int>(offset);
//	std::vector<size_t> ids;
//	std::vector<int> process_result = postprocessObj->RunCommon(predict_result_3d[0][0],ids,expected);
//	final_result = postprocessObj->UploadResults();
//	std::cout << "num of good: " << final_result["good"] << " total num: " << final_result["total"] << std::endl;
//	
//	postprocess::PostProcessBase<int>* postprocessObj1;
//	postprocessObj1 = new postprocess::PostProcArgMax<int>(offset);
//	std::vector<int> process_result_argmax = postprocessObj1->RunArgMax(predict_result_3d[0], ids, expected);
//	final_result1 = postprocessObj1->UploadResults();
//	std::cout << "num of good: " << final_result1["good"] << " total num: " << final_result1["total"] << std::endl;
//
//	delete postprocessObj;
//	delete postprocessObj1;
//
//
//	//test PostProcCoco, 构造一组coco预测结果数据
//std::vector <float*> result_coco;
//float detection_nums[2] = { 1,2 };
//float detection_boxes[12] = { 1,2,2,1,3,4,4,3,5,6,6,5 };
//float detection_scores[3] = { 8.8, 9.9, 11.1 };
//float detection_classes[3] = { 1,4,3 };
//result_coco.push_back(detection_nums);
//result_coco.push_back(detection_boxes);
//result_coco.push_back(detection_scores);
//result_coco.push_back(detection_classes);
//
//std::vector<std::vector<float>> expectedCoco;
//expectedCoco.push_back({ 1 });
//expectedCoco.push_back({ 2, 4 });
//
//std::vector<size_t> ids_test = { 1,2,3 };
//
//
//postprocess::PostProcessBase<float>* postprocessObj2;
//postprocessObj2 = new postprocess::PostProcessCoco<float>();
//std::vector<std::vector<std::vector<float>>> postprocresultCoco;
//postprocresultCoco = postprocessObj2->RunCoco(result_coco, ids_test, expectedCoco);
//final_result2 = postprocessObj2->UploadResults();
//std::cout << "num of good: " << final_result2["good"] << " total num: " << final_result2["total"] << std::endl;
//
//postprocessObj2->AddResultsCoco(postprocresultCoco);
//
//std::vector<std::vector<std::vector<std::vector<float>>>> finalresultCoco;
//finalresultCoco = postprocessObj2->UploadResultsCoco();
//for (size_t i = 0; i < finalresultCoco.size(); i++)
//{
//	for (size_t j = 0; j < finalresultCoco[i].size(); j++)
//	{
//		for (size_t m = 0; m < finalresultCoco[i][j].size(); m++)
//		{
//			for (size_t n = 0; n < finalresultCoco[i][j][m].size(); n++)
//			{
//				std::cout << finalresultCoco[i][j][m][n] << " , ";
//			}
//			std::cout << std::endl;
//		}
//	}
//}
//delete postprocessObj2;
//
////test PostProcCocoPt
//postprocess::PostProcessBase<float>* postprocessObj3;
//postprocessObj3 = new postprocess::PostProcessCocoPt<float>(false, 9.91);
//std::vector<std::vector<std::vector<float>>> postprocresultCocoPt;
//postprocresultCocoPt = postprocessObj3->RunCoco(result_coco, ids_test, expectedCoco);
//final_result3 = postprocessObj3->UploadResults();
//std::cout << "num of good: " << final_result3["good"] << " total num: " << final_result3["total"] << std::endl;
//
//postprocessObj3->AddResultsCoco(postprocresultCocoPt);
//
//std::vector<std::vector<std::vector<std::vector<float>>>> finalresultCocoPt;
//finalresultCocoPt = postprocessObj3->UploadResultsCoco();
//for (size_t i = 0; i < finalresultCocoPt.size(); i++)
//{
//	for (size_t j = 0; j < finalresultCocoPt[i].size(); j++)
//	{
//		for (size_t m = 0; m < finalresultCocoPt[i][j].size(); m++)
//		{
//			for (size_t n = 0; n < finalresultCocoPt[i][j][m].size(); n++)
//			{
//				std::cout << finalresultCocoPt[i][j][m][n] << " , ";
//			}
//			std::cout << std::endl;
//		}
//	}
//}
//delete postprocessObj3;
//
//}
//
