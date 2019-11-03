#include <iostream>
#include <functional>
#include <chrono>
#include <queue>
#include <unistd.h>

#include "schedule/init_node.h"
#include "schedule/image_node.h"
#include "schedule/batch_merge_node.h"
#include "schedule/batch_split_node.h"
#include "schedule/gpu_schedule_node.h"
#include "schedule/post_process_node.h"
#include "schedule/data_set.h"
#include "common/logging.h"
#include "schedule/schedule.h"

using namespace std;
using namespace schedule;


//class TestClass1 {
//public:
//	void fun1(int var1) {
//		cout << 1 << endl;
//	}
//};
//
//using fun = std::function<void(int)>;
////typedef void (*fun) (int);
//void fun1(int var1) {
//	cout << 1 << endl;
//}
//
//class TestClass2 {
//public:
//	void fun2(fun f) {
//		f(0);
//	}
//	~TestClass2() {
//		cout << "destruct..." << endl;
//	}
//};
//
//auto TestClass2Deleter = [](TestClass2* p_cls) {
//	delete[] p_cls;
//};
//
//void funv(vector<int> v) {
//	vector<int> v2{ 0 };
//	cout << &v[0] << endl;
//	cout << &v2[0] << endl;
//	v2.swap(v);
//	cout << &v[0] << endl;
//	cout << &v2[0] << endl;
//}

//void ReportLatencyResultsCb(ClientData, const int64_t*,
//	size_t) {
//
//}

//int main()
//{
//	//queue<int> q;
//	//cout << q.size() << endl;
//	//for (size_t i = 0; i < 10; i++) {
//	//	q.push(1);
//	//	sleep(1);
//	//}
//	//cout << q.size() << endl;
//	//{
//	//	TestClass2* cls = new TestClass2[1];
//	//	shared_ptr<TestClass2> sp1(cls, TestClass2Deleter);
//	//	shared_ptr<TestClass2> sp2(cls, TestClass2Deleter);
//	//}
//
////	size_t test_round = 1;
////	for (size_t i = 0; i < test_round; i++) {
//	Schedule* schedule = Schedule::GetSchedule();
//
//	mp::MLPerfSettings& settings = schedule->get_m_settings();
//	TestSettings& test_settings = schedule->GetInferenceSettings();
//
//#ifdef WIN32
//	string data_dir = "D:\\work\\code\\inference\\v0.5\\classification_and_detection\\preprocessed\\imagenet\\NHWC";
//	string cache_dir = "D:\\work\\code\\inference\\v0.5\\classification_and_detection\\preprocessed\\imagenet\\NHWC";
//	string model_path = "D:\\resnet50_v1_tf\\resnet50_v1.pb";
//#else
//	string data_dir = "/data";
//	string cache_dir = "/root/inference/v0.5/classification_and_detection/preprocessed/imagenet/NHWC";
//	string model_path = "/root/resnet50_v1.pb";
//#endif	
//	string data_set = "imagenet";
//	string profile_name = "resnet50-tf";
//	string backend = "tensorflow";
//	bool accuracy = false;
//	vector<string> inputs{ "input_tensor:0" };
//	vector<string> outputs{ "ArgMax:0" };
//	vector<string> image_list;
//	vector<float> label_list;
//	image_list.reserve(10);
//	label_list.reserve(10);
//	image_list.push_back("ILSVRC2012_val_00000001.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000002.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000003.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000004.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000005.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000006.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000007.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000008.JPEG");
//	image_list.push_back("ILSVRC2012_val_00000009.JPEG");
//	image_list.push_back("ILSVRC2012_val_000000010.JPEG");
//	label_list.push_back(65);
//	label_list.push_back(970);
//	label_list.push_back(230);
//	label_list.push_back(809);
//	label_list.push_back(516);
//	label_list.push_back(57);
//	label_list.push_back(334);
//	label_list.push_back(415);
//	label_list.push_back(674);
//	label_list.push_back(332);
//	size_t min_query_count = 1;
//	size_t max_query_count = 1;
//
//	settings.set_m_data_set(data_set);
//	settings.set_m_data_dir(data_dir);
//	settings.set_m_cache_dir(cache_dir);
//	settings.set_m_model_path(model_path);
//	settings.set_m_profile_name(profile_name);
//	settings.set_m_backend(backend);
//	settings.set_m_accuracy(accuracy);
//	settings.set_m_inputs(inputs);
//	settings.set_m_outputs(outputs);
//	settings.set_m_image_list(image_list);
//	settings.set_m_label_list(label_list);
//	settings.get_m_test_settings().min_query_count = min_query_count;
//	settings.get_m_test_settings().max_query_count = max_query_count;
//	settings.get_m_test_settings().scenario = mlperf::TestScenario::Offline;
//
//	schedule->InitSchedule();
//
//	schedule->InitMLPerf(ReportLatencyResultsCb);
//
//	schedule->StartTest(test_settings,
//		data_set,
//		data_dir,
//		cache_dir,
//		model_path,
//		profile_name,
//		backend,
//		accuracy,
//		inputs,
//		outputs,
//		image_list,
//		label_list);
//
//	schedule->FinalizeMLPerf();
//
//	cout << endl;
//
//	//		Schedule::DestroySchedule();
//	//	}
//
//}
