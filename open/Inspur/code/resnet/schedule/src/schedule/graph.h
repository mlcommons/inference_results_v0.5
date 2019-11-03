# pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/config.h"
#include "../config/parser.h"
#include "../config/input_file.h"

using namespace std;


namespace schedule {

	class ScheduleNode : public Node {
	public:
		ScheduleNode(string name, string type, vector<string> bottom, vector<string> top)
			: Node(name, type, bottom, top) {};
		virtual ~ScheduleNode() {};

		virtual void Init() {};
		virtual void Run() {};
		virtual void Finalize() {};
		virtual void* GetFreeQueue() { return nullptr; };
		virtual void* GetFullQueue() { return nullptr; };
		ScheduleNode* GetBottomNode();
		ScheduleNode* GetTopNode();
	};

	class Graph {
		CREATE_SIMPLE_ATTR_GET(m_conf, Config)
		// Nodes after optimization.
		CREATE_SIMPLE_ATTR_GET(m_nodes, vector<shared_ptr<ScheduleNode>>)

	public:
		Graph() {};
		~Graph() {};

		void ReadConfigFile(string path);
		void Init();
		void Run();
		void Stop();
		void Optimize();
		vector<ScheduleNode*> GetNodesByType(string type);
		vector<ScheduleNode*> GetNodesByTypeAll(string type);
	};

	class Optimizer {
		CREATE_SIMPLE_ATTR_GET(m_p_graph, Graph*)

	public:
		Optimizer(Graph* p_graph) {
			m_p_graph = p_graph;
		};
		virtual ~Optimizer() {
		};

		virtual void Optimize() = 0;
	};

	class SingleStreamOptimizer : public Optimizer {
	public:
		SingleStreamOptimizer(Graph* p_graph): Optimizer(p_graph) {};
		~SingleStreamOptimizer() {};

		void Optimize();
	};

	class MultiStreamOptimizer : public Optimizer {
	public:
		MultiStreamOptimizer(Graph* p_graph) : Optimizer(p_graph) {};
		~MultiStreamOptimizer() {}

		void Optimize();
	};

	class ServerOptimizer : public Optimizer {
	public:
		ServerOptimizer(Graph* p_graph) : Optimizer(p_graph) {};
		~ServerOptimizer() {}

		void Optimize();
	};

	class OfflineOptimizer : public Optimizer {
	public:
		OfflineOptimizer(Graph* p_graph) : Optimizer(p_graph) {};
		~OfflineOptimizer() {}

		void Optimize();
	};

}