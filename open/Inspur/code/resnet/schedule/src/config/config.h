#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <map>
#include <string>
#include <vector>
#include <memory>

using namespace std;


namespace schedule {

	class Node;
	class Config {
	public:
		Config();
		~Config();

		string GetName() const;
		string GetInput() const;
		size_t GetBatchSize() const;
		size_t GetNumChannels() const;
		size_t GetInputHeight() const;
		size_t GetInputWidth() const;
		vector<shared_ptr<Node>>* GetNodes();
		map<string, shared_ptr<Node>>* GetNameNodeMap();

		void SetName(string name);
		void SetInput(string input);
		void SetBatchSize(size_t batch_size);
		void SetNumChannels(size_t num_channels);
		void SetInputHeight(size_t input_height);
		void SetInputWidth(size_t input_width);
	private:
		string m_name;
		string m_input;
		size_t m_batch_size;
		size_t m_num_channels;
		size_t m_input_height;
		size_t m_input_width;
		vector<shared_ptr<Node>> m_p_nodes;
		map<string, shared_ptr<Node>> m_name_node_map;
	};

}


#endif // !__CONFIG_H__
