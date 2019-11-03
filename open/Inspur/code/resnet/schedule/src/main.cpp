#include <cstdio>
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


void ReportLatencyResultsCb(vector<int64_t>) {

}

class TestClass1 {
public:
	void fun1(int var1) {
		cout << 1 << endl;
	}

	friend ostream& operator<<(ostream& stream, const TestClass1& c1) {
		stream << c1.a1 << c1.a2 << c1.a3 << endl;
		return stream;
	}

private:
	int a1;
	char a2[10];
	string a3;
	shared_ptr<vector<size_t>> data;
};

//using fun = std::function<void(int)>;
typedef void (*fun) (int);
void fun1(int var1) {
	cout << 1 << endl;
}

class TestClass2 {
public:
	void fun2(fun f) {
		f(0);
	}
	~TestClass2() {
		cout << "destruct..." << endl;
	}
};

int main()
{
	Schedule* schedule = Schedule::GetSchedule();

	mp::MLPerfSettings& settings = schedule->get_m_settings();
	TestSettings& test_settings = schedule->GetInferenceSettings();

	string conf_path = "/root/schedule/mlperf_inference_schedule.prototxt";

	// resnet50-tf
	//string data_dir = "/data/dataset-imagenet-ilsvrc2012-val";
	//string cache_dir = "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess/preprocessed/imagenet/NCHW";
	//string model_path = "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/resnet50_v1_1.trt";
	//string data_set = "imagenet";
	//string profile_name = "resnet50-tf";

	// mobilenet-tf
	//string data_dir = "/data/dataset-imagenet-ilsvrc2012-val";
	//string cache_dir = "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess/preprocessed/imagenet_mobilenet/NCHW";
	//string model_path = "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/mobilenet_v1_1.0_224_1.trt";
	//string data_set = "imagenet_mobilenet";
	//string profile_name = "mobilenet-tf";

	// ssd-mobilenet-tf
	string data_dir = "/data/dataset-coco-2017-val";
	string cache_dir = "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess/preprocessed/coco-300/NCHW/val2017";
	string model_path = "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/ssd_mobilenet_1.trt";
	string data_set = "coco-300";
	string profile_name = "ssd-mobilenet-tf";

	string backend = "trt";
	bool accuracy = false;
	vector<string> inputs{ "input_tensor:0" };
	vector<string> outputs{ "ArgMax:0" };
	vector<string> image_list;
	vector< vector<float>> label_list;
	image_list.reserve(50000);
	label_list.reserve(50000);
	for (auto i = 0; i < 50000; i++)
	{
		char image_name[100];
		//sprintf(image_name, "ILSVRC2012_val_00000001.JPEG");
		sprintf(image_name, "000000119445.jpg");
		string image_name_s = image_name;
		image_list.push_back(image_name_s);
		vector<float> labels;
		labels.push_back(65);
		label_list.push_back(labels);
	}

//	size_t min_query_count = 512;
//	size_t max_query_count = 512;

	settings.set_m_data_set(data_set);
	settings.set_m_data_dir(data_dir);
	settings.set_m_cache_dir(cache_dir);
	settings.set_m_model_path(model_path);
	settings.set_m_profile_name(profile_name);
	settings.set_m_backend(backend);
	settings.set_m_accuracy(accuracy);
	settings.set_m_inputs(inputs);
	settings.set_m_outputs(outputs);
	settings.set_m_image_list(image_list);
	settings.set_m_label_list(label_list);
//	settings.get_m_test_settings().min_query_count = min_query_count;
//	settings.get_m_test_settings().max_query_count = max_query_count;
//	settings.get_m_test_settings().multi_stream_samples_per_query = 64;
	settings.get_m_test_settings().scenario = mlperf::TestScenario::SingleStream;
	schedule->InitSchedule(conf_path,
		test_settings,
		data_set,
		data_dir,
		cache_dir,
		model_path,
		profile_name,
		backend,
		accuracy,
		inputs,
		outputs,
		image_list,
		label_list);

	schedule->InitMLPerf(ReportLatencyResultsCb);

	schedule->StartTest();

	schedule->UploadResults();

	schedule->FinalizeMLPerf();

	cout << endl;
}
