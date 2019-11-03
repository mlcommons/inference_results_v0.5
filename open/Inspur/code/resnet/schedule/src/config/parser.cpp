#include <memory>

#include "node.h"
#include "config.h"
#include "parser.h"
#include "input_file.h"
#include "../schedule/init_node.h"
#include "../schedule/image_node.h"
#include "../schedule/batch_merge_node.h"
#include "../schedule/batch_split_node.h"
#include "../schedule/gpu_schedule_node.h"
#include "../schedule/memory_copy_node.h"
#include "../schedule/inference_node.h"
#include "../schedule/post_process_node.h"

using namespace std;


namespace schedule {

	Parser::Parser(InputFile* pfile, Config* pconf) {
		m_pfile = pfile;
		m_pconf = pconf;
	}

	Parser::~Parser() {

	}

	void Parser::Run() {
		// common config
		string name = "";
		string input = "";
		vector<size_t> input_dim;
		size_t batch_size = 0;
		size_t num_channels = 0;
		size_t input_height = 0;
		size_t input_width = 0;

		map<string, string*> key_value;
		key_value["name:"] = &name;
		key_value["input:"] = &input;
		this->ParseLines<string>(key_value, "", true);

		map<string, vector<size_t>*> key_value2;
		key_value2["input_dim:"] = &input_dim;
		this->ParseLinesV<size_t>(key_value2, "", true);

		m_pconf->SetName(name);
		m_pconf->SetInput(input);

		if (input_dim.size() >= 4) {
			batch_size = input_dim[0];
			num_channels = input_dim[1];
			input_height = input_dim[2];
			input_width = input_dim[3];

			m_pconf->SetBatchSize(batch_size);
			m_pconf->SetNumChannels(num_channels);
			m_pconf->SetInputHeight(input_height);
			m_pconf->SetInputWidth(input_width);
		}

		// nodes
		vector<shared_ptr<Node>>* p_nodes = this->m_pconf->GetNodes();
		map<string, shared_ptr<Node>>* p_name_node_map = this->m_pconf->GetNameNodeMap();

		while (true) {
			string line = m_pfile->ReadLine();
			if (line == "")
				break;
			if (line == "node {") {
				NodeParser lp(this->m_pfile, this->m_pconf);
				Node node = lp.Run();
				shared_ptr<Node> pn;
				if (node.GetType() == "ImageData") {
					ImageDataNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<ImageDataNode>(p.Run(node));
				}
				else if (node.GetType() == "Convolution") {
					ConvolutionNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<ConvolutionNode>(p.Run(node));
				}
				else if (node.GetType() == "Pooling") {
					PoolingNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<PoolingNode>(p.Run(node));
				}
				else if (node.GetType() == "BatchNorm") {
					BatchNormNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<BatchNormNode>(p.Run(node));
				}
				else if (node.GetType() == "ReLU") {
					ReluNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<ReluNode>(p.Run(node));
				}
				else if (node.GetType() == "InnerProduct") {
					InnerProductNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<InnerProductNode>(p.Run(node));
				}
				else if (node.GetType() == "Eltwise") {
					EltwiseNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<EltwiseNode>(p.Run(node));
				}
				else if (node.GetType() == "Softmax") {
					SoftmaxNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<SoftmaxNode>(p.Run(node));
				}
				else if (node.GetType() == "Accuracy") {
					AccuracyNodeParser p(this->m_pfile, this->m_pconf);
					pn = make_shared<AccuracyNode>(p.Run(node));
				}
				// For MLPerf
				else if (node.GetType() == "Init") {
					schedule::InitNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "Image") {
					schedule::ImageNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "BatchMerge") {
					schedule::BatchMergeNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "BatchSplit") {
					schedule::BatchSplitNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "GpuSchedule") {
					schedule::GpuScheduleNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "MemoryCopy") {
					schedule::MemoryCopyNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "Inference") {
					schedule::InferenceNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				else if (node.GetType() == "PostProcess") {
					schedule::PostProcessNodeParser p(this->m_pfile, this->m_pconf);
					pn = p.Run(node);
				}
				p_nodes->push_back(pn);
				(*p_name_node_map)[node.GetName()] = pn;
			}
		}
	}

	NodeParser::NodeParser(InputFile* pfile, Config* pconf) : Parser(pfile, pconf) {

	}

	NodeParser::~NodeParser() {

	}

	Node NodeParser::ParseCommon() {
		// common config
		string name = "";
		string type = "";
		vector<string> bottom;
		vector<string> top;

		map<string, string*> key_value;
		key_value["name:"] = &name;
		key_value["type:"] = &type;
		this->ParseLines<string>(key_value, "", true);

		map<string, vector<string>*> key_value2;
		key_value2["bottom:"] = &bottom;
		key_value2["top:"] = &top;
		this->ParseLinesV<string>(key_value2, "", true);

		return Node(name, type, bottom, top);
	}

	WeightFiller NodeParser::ParseWeightFiller() {
		string type = "";

		map<string, string*> key_value;
		key_value["type:"] = &type;
		this->ParseLines<string>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return WeightFiller(type);
	}

	Include NodeParser::ParseInclude() {
		string line = "";
		stringstream word;
		string key = "";
		string phase = "";

		line = m_pfile->ReadLine();
		if (line == "") {
			throw "read incude param fail";
		}
		word = stringstream(line);
		word >> key;
		word >> key;
		word >> key;
		word >> phase;
		phase = common::strip(phase, "\"");
		return Include(phase);
	}

	Node NodeParser::Run() {
		return this->ParseCommon();
	}

	ImageDataNodeParser::ImageDataNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	ImageDataNodeParser::~ImageDataNodeParser() {

	}

	ImageDataNode ImageDataNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "image_data_param {")
			throw "read image_data_param fail";
		ImageDataParam idp = this->ParseImageDataParam();

		line = m_pfile->ReadLine();
		if (line != "transform_param {")
			throw "read transform_param fail";
		TransformParam tp = this->ParseTransformParam();

		Include i = this->ParseInclude();
		return ImageDataNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), idp, tp, i);
	}

	ImageDataParam ImageDataNodeParser::ParseImageDataParam() {
		string root_folder = "";
		string source = "";
		size_t batch_size = 0;

		map<string, string*> key_value;
		key_value["root_folder:"] = &root_folder;
		key_value["source:"] = &source;
		this->ParseLines<string>(key_value, "", true);

		map<string, size_t*> key_value2;
		key_value2["batch_size:"] = &batch_size;
		this->ParseLines<size_t>(key_value2, "", true);

		string _ = m_pfile->ReadLine();

		return ImageDataParam(root_folder, source, batch_size);
	}

	TransformParam ImageDataNodeParser::ParseTransformParam() {
		size_t crop_size = 0;
		bool mirror = false;
		float scale = 0;

		map<string, size_t*> key_value;
		key_value["crop_size:"] = &crop_size;
		this->ParseLines<size_t>(key_value, "", true);

		map<string, bool*> key_value2;
		key_value2["mirror:"] = &mirror;
		this->ParseLines<bool>(key_value2, "", true);

		map<string, float*> key_value3;
		key_value3["scale:"] = &scale;
		this->ParseLines<float>(key_value3, "", true);

		string _ = m_pfile->ReadLine();

		return TransformParam(crop_size, mirror, scale);
	}

	ConvolutionNodeParser::ConvolutionNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	ConvolutionNodeParser::~ConvolutionNodeParser() {

	}

	ConvolutionNode ConvolutionNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "convolution_param {")
			throw "read convolution_param fail";
		ConvolutionParam cp = this->ParseConvolutionParam();

		string _ = m_pfile->ReadLine();

		return ConvolutionNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), cp);
	}

	ConvolutionParam ConvolutionNodeParser::ParseConvolutionParam() {
		size_t num_output = 0;
		size_t kernel_size = 0;
		size_t pad = 0;
		size_t stride = 1;
		bool bias_term = false;

		map<string, size_t*> key_value;
		key_value["num_output:"] = &num_output;
		key_value["kernel_size:"] = &kernel_size;
		key_value["pad:"] = &pad;
		key_value["stride:"] = &stride;
		this->ParseLines<size_t>(key_value, "", true);

		string line = m_pfile->ReadLine();
		if (line != "weight_filler {")
			throw "read weight_filler fail";
		WeightFiller filler = this->ParseWeightFiller();

		map<string, bool*> key_value2;
		key_value2["bias_term:"] = &bias_term;
		this->ParseLines<bool>(key_value2, "", true);

		string _ = m_pfile->ReadLine();

		return ConvolutionParam(num_output, kernel_size, pad, stride, filler, bias_term);
	}

	PoolingNodeParser::PoolingNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	PoolingNodeParser::~PoolingNodeParser() {

	}

	PoolingNode PoolingNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "pooling_param {")
			throw "read pooling_param fail";
		PoolingParam pp = this->ParsePoolingParam();

		string _ = m_pfile->ReadLine();

		return PoolingNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), pp);
	}

	PoolingParam PoolingNodeParser::ParsePoolingParam() {
		string pool = "";
		size_t kernel_size = 0;
		size_t stride = 0;

		map<string, string*> key_value;
		key_value["pool:"] = &pool;
		this->ParseLines<string>(key_value, "", true);

		map<string, size_t*> key_value2;
		key_value2["kernel_size:"] = &kernel_size;
		key_value2["stride:"] = &stride;
		this->ParseLines<size_t>(key_value2, "", true);

		string _ = m_pfile->ReadLine();

		return PoolingParam(pool, kernel_size, stride);
	}

	BatchNormNodeParser::BatchNormNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	BatchNormNodeParser::~BatchNormNodeParser() {

	}

	BatchNormNode BatchNormNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "batch_norm_param {")
			throw "read batch_norm_param fail";
		BatchNormParam bnp = this->ParseBatchNormParam();

		string _ = m_pfile->ReadLine();

		return BatchNormNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), bnp);
	}

	BatchNormParam BatchNormNodeParser::ParseBatchNormParam() {
		float moving_average_fraction = 0;
		float eps = 0;
		bool scale_bias = false;

		map<string, float*> key_value;
		key_value["moving_average_fraction:"] = &moving_average_fraction;
		key_value["eps:"] = &eps;
		this->ParseLines<float>(key_value, "", true);

		map<string, bool*> key_value2;
		key_value2["scale_bias:"] = &scale_bias;
		this->ParseLines<bool>(key_value2, "", true);

		string _ = m_pfile->ReadLine();

		return BatchNormParam(moving_average_fraction, eps, scale_bias);
	}

	ReluNodeParser::ReluNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	ReluNodeParser::~ReluNodeParser() {

	}

	ReluNode ReluNodeParser::Run(Node node) {
		string _ = m_pfile->ReadLine();
		return ReluNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop());
	}

	InnerProductNodeParser::InnerProductNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	InnerProductNodeParser::~InnerProductNodeParser() {

	}

	InnerProductNode InnerProductNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "inner_product_param {")
			throw "read inner_product_param fail";
		InnerProductParam ipp = this->ParseInnerProductParam();

		string _ = m_pfile->ReadLine();

		return InnerProductNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), ipp);
	}

	InnerProductParam InnerProductNodeParser::ParseInnerProductParam() {
		size_t num_output = 0;

		map<string, size_t*> key_value;
		key_value["num_output:"] = &num_output;
		this->ParseLines<size_t>(key_value, "", true);

		string line = m_pfile->ReadLine();
		if (line != "weight_filler {")
			throw "read weight_filler fail";
		WeightFiller wf = this->ParseWeightFiller();

		line = m_pfile->ReadLine();
		if (line != "bias_filler {")
			throw "read bias_filler fail";
		BiasFiller bf = this->ParseBiasFiller();

		string _ = m_pfile->ReadLine();

		return InnerProductParam(num_output, wf, bf);
	}

	BiasFiller InnerProductNodeParser::ParseBiasFiller() {
		string type = "";
		float value = 0;

		map<string, string*> key_value;
		key_value["type:"] = &type;
		this->ParseLines<string>(key_value, "", true);

		map<string, float*> key_value2;
		key_value2["value:"] = &value;
		this->ParseLines<float>(key_value2, "", true);

		string _ = m_pfile->ReadLine();

		return BiasFiller(type, value);
	}

	EltwiseNodeParser::EltwiseNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	EltwiseNodeParser::~EltwiseNodeParser() {

	}

	EltwiseNode EltwiseNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "eltwise_param {")
			throw "read eltwise_param fail";
		EltwiseParam ep = this->ParseEltwiseParam();

		string _ = m_pfile->ReadLine();

		return EltwiseNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), ep);
	}

	EltwiseParam EltwiseNodeParser::ParseEltwiseParam() {
		string operation;

		map<string, string*> key_value;
		key_value["operation:"] = &operation;
		this->ParseLines<string>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return EltwiseParam(operation);
	}

	SoftmaxNodeParser::SoftmaxNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	SoftmaxNodeParser::~SoftmaxNodeParser() {

	}

	SoftmaxNode SoftmaxNodeParser::Run(Node node) {
		string _ = m_pfile->ReadLine();
		return SoftmaxNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop());
	}

	AccuracyNodeParser::AccuracyNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {

	}

	AccuracyNodeParser::~AccuracyNodeParser() {

	}

	AccuracyNode AccuracyNodeParser::Run(Node node) {
		AccuracyParam ap = this->ParseAccuracyParam();
		Include i = this->ParseInclude();

		string _ = m_pfile->ReadLine();

		return AccuracyNode(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), ap, i);
	}

	AccuracyParam AccuracyNodeParser::ParseAccuracyParam() {
		string line = "";
		stringstream word;
		string key = "";
		size_t top_k = 0;

		line = m_pfile->ReadLine();
		if (line == "") {
			throw "read accuracy_param fail";
		}
		word = stringstream(line);
		word >> key;
		word >> key;
		word >> key;
		word >> top_k;
		return AccuracyParam(top_k);
	}

}