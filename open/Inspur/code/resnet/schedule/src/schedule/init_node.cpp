#include "../config/input_file.h"
#include "init_node.h"


namespace schedule {

	shared_ptr<InitNode> InitNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "init_param {")
			throw "read init_param fail";
		InitParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<InitNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	InitParam InitNodeParser::ParseParam() {
		size_t queue_depth = 100;
		size_t batch_size = 20000;

		map<string, size_t*> key_value;
		key_value["queue_depth:"] = &queue_depth;
		key_value["batch_size:"] = &batch_size;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return InitParam(queue_depth, batch_size);
	}

	void InitNode::Init() {
		for (size_t i = 0; i < m_param.get_m_queue_depth(); i++) {
			shared_ptr<Batch<QuerySample>> bsp = make_shared<Batch<QuerySample>>();
			m_queue_free.Push(bsp);
		}
	}

	void InitNode::Finalize() {
		m_queue_free.Clear();
		m_queue_full.Clear();
	}

}