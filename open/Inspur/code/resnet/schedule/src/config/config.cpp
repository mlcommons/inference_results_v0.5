#include "config.h"

using namespace std;


namespace schedule {

	Config::Config() {
		m_name = "";
		m_input = "";
		m_batch_size = 1;
		m_num_channels = 3;
		m_input_height = 224;
		m_input_width = 224;
	}

	Config::~Config() {
	}

	string Config::GetName() const {
		return m_name;
	}

	string Config::GetInput() const {
		return m_input;
	}

	size_t Config::GetBatchSize() const {
		return m_batch_size;
	}

	size_t Config::GetNumChannels() const {
		return m_num_channels;
	}

	size_t Config::GetInputHeight() const {
		return m_input_height;
	}

	size_t Config::GetInputWidth() const {
		return m_input_width;
	}

	vector<shared_ptr<Node>>* Config::GetNodes() {
		return &m_p_nodes;
	}

	map<string, shared_ptr<Node>>* Config::GetNameNodeMap() {
		return &m_name_node_map;
	}

	void Config::SetName(string name) {
		m_name = name;
	}

	void Config::SetInput(string input) {
		m_input = input;
	}

	void Config::SetBatchSize(size_t batch_size) {
		m_batch_size = batch_size;
	}

	void Config::SetNumChannels(size_t num_channels) {
		m_num_channels = num_channels;
	}

	void Config::SetInputHeight(size_t input_height) {
		m_input_height = input_height;
	}

	void Config::SetInputWidth(size_t input_width) {
		m_input_width = input_width;
	}

}