#include "init_node.h"
#include "image_node.h"
#include "batch_merge_node.h"
#include "batch_split_node.h"
#include "gpu_schedule_node.h"
#include "memory_copy_node.h"
#include "inference_node.h"
#include "post_process_node.h"
#include "graph.h"


namespace schedule {

	ScheduleNode* ScheduleNode::GetBottomNode() {
		auto& bottoms = GetBottom();
		if (bottoms.empty())
			return nullptr;
		string bottom_name = bottoms[0];
		auto n_n_map = Schedule::GetSchedule()->get_m_graph().get_m_conf().GetNameNodeMap();
		if (n_n_map->count(bottom_name) <= 0)
			return nullptr;
		shared_ptr<Node> bottom = (*n_n_map)[bottom_name];
		return dynamic_cast<ScheduleNode*>(bottom.get());
	}

	// Caution: nodes should be after optimizing.
	ScheduleNode* ScheduleNode::GetTopNode() {
		auto& p_nodes = Schedule::GetSchedule()->get_m_graph().get_m_nodes();
		for (auto& p_node : p_nodes) {
			auto& bottoms = p_node->GetBottom();
			if (bottoms.empty())
				continue;
			string bottom_name = bottoms[0];
			if (bottom_name == GetName()) {
				return p_node.get();
			}
		}
		return nullptr;
	}

	void Graph::ReadConfigFile(string path) {
		InputFile file(path);
		Parser parser(&file, &m_conf);
		parser.Run();
	}

	void Graph::Init() {
		for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
			(*it)->Init();
		}
	}

	void Graph::Run() {
		for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
			(*it)->Run();
		}
	}

	void Graph::Stop() {
		for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
			(*it)->Finalize();
		}
	}

	void Graph::Optimize() {
		TestScenario s = Schedule::GetSchedule()->get_m_settings().get_m_test_settings().scenario;
		
		if (s == TestScenario::SingleStream) {
			SingleStreamOptimizer(this).Optimize();
		}
		else if (s == TestScenario::MultiStream) {
			MultiStreamOptimizer(this).Optimize();
		}
		else if (s == TestScenario::Server) {
			ServerOptimizer(this).Optimize();
		}
		else if (s == TestScenario::Offline) {
			OfflineOptimizer(this).Optimize();
		}
	}

	vector<ScheduleNode*> Graph::GetNodesByType(string type) {
		vector<ScheduleNode*> results;
		for (auto it = m_nodes.begin(); it != m_nodes.end(); it++) {
			string t = (*it)->GetType();
			if (t == type) {
				results.push_back(dynamic_cast<ScheduleNode*>(it->get()));
			}
		}
		return results;
	}

	vector<ScheduleNode*> Graph::GetNodesByTypeAll(string type) {
		vector<ScheduleNode*> results;
		auto nodes = m_conf.GetNodes();
		for (auto it = nodes->begin(); it != nodes->end(); it++) {
			string t = (*it)->GetType();
			if (t == type) {
				results.push_back(dynamic_cast<ScheduleNode*>(it->get()));
			}
		}
		return results;
	}

	void SingleStreamOptimizer::Optimize() {
		auto& nodes_optimized = m_p_graph->get_m_nodes();
		nodes_optimized.clear();
		auto nodes = m_p_graph->get_m_conf().GetNodes();

		for (auto it = nodes->begin(); it != nodes->end(); it++) {
			string type = (*it)->GetType();
			if (type == "Image" || type == "BatchMerge" || type == "BatchSplit") {
				continue;
			}
			else if (type == "GpuSchedule") {
				GpuScheduleNode* p_node = dynamic_cast<GpuScheduleNode*>((*it).get());
				InnerThread* p_thread = dynamic_cast<InnerThread*>(p_node);
				GpuScheduleParam& param = p_node->get_m_param();
				param.set_m_thread_num(1);
				param.set_m_batch_size(1);
				p_thread->set_m_thread_num(1);
			}
			else if (type == "MemoryCopy") {
				MemoryCopyNode* p_node = dynamic_cast<MemoryCopyNode*>((*it).get());
				InnerThread* p_thread = dynamic_cast<InnerThread*>(p_node);
				MemoryCopyParam& param = p_node->get_m_param();
				param.set_m_thread_num(1);
				param.set_m_batch_size(1);
				p_thread->set_m_thread_num(1);
			}
			else if (type == "Inference") {
				InferenceNode* p_node = dynamic_cast<InferenceNode*>((*it).get());
				InnerThread* p_thread = dynamic_cast<InnerThread*>(p_node);
				InferenceParam& param = p_node->get_m_param();
				param.set_m_batch_size(1);
				param.set_m_gpu_engine_num(1);
				p_thread->set_m_thread_num(1);
			}
			else if (type == "PostProcess") {
				PostProcessNode* p_node = dynamic_cast<PostProcessNode*>((*it).get());
				InnerThread* p_thread = dynamic_cast<InnerThread*>(p_node);
				PostProcessParam& param = dynamic_cast<PostProcessNode*>((*it).get())->get_m_param();
				param.set_m_batch_size(1);
				param.set_m_thread_num(1);
				p_thread->set_m_thread_num(1);
			}

			if (nodes_optimized.size() > 0) {
				shared_ptr<Node> p_node = (*it);
				p_node->GetBottom()[0] = nodes_optimized[nodes_optimized.size() - 1]->GetName();
			}

			nodes_optimized.push_back(shared_ptr<ScheduleNode>(dynamic_cast<ScheduleNode*>(it->get())));
		}
	}

	void MultiStreamOptimizer::Optimize() {
		auto& nodes_optimized = m_p_graph->get_m_nodes();
		nodes_optimized.clear();
		auto nodes = m_p_graph->get_m_conf().GetNodes();
		//auto inference_node = dynamic_cast<InferenceNode*>(m_p_graph->GetNodesByTypeAll("Inference")[0]);
		//auto batch_size = inference_node->get_m_param().get_m_batch_size();

		for (auto it = nodes->begin(); it != nodes->end(); it++) {
			string type = (*it)->GetType();
			if (type == "Image" || type == "BatchMerge") {
				continue;
			}
			//else if (type == "BatchSplit") {
			//	BatchSplitNode* p_node = dynamic_cast<BatchSplitNode*>((*it).get());
			//	BatchSplitParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "GpuSchedule") {
			//	GpuScheduleNode* p_node = dynamic_cast<GpuScheduleNode*>((*it).get());
			//	GpuScheduleParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "MemoryCopy") {
			//	MemoryCopyNode* p_node = dynamic_cast<MemoryCopyNode*>((*it).get());
			//	MemoryCopyParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "Inference") {
			//	InferenceNode* p_node = dynamic_cast<InferenceNode*>((*it).get());
			//	InferenceParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "PostProcess") {
			//	PostProcessNode* p_node = dynamic_cast<PostProcessNode*>((*it).get());
			//	PostProcessParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}

			if (nodes_optimized.size() > 0) {
				shared_ptr<Node> p_node = (*it);
				p_node->GetBottom()[0] = nodes_optimized[nodes_optimized.size() - 1]->GetName();
			}

			nodes_optimized.push_back(shared_ptr<ScheduleNode>(dynamic_cast<ScheduleNode*>(it->get())));
		}
	}

	void ServerOptimizer::Optimize() {
		auto& nodes_optimized = m_p_graph->get_m_nodes();
		nodes_optimized.clear();
		auto nodes = m_p_graph->get_m_conf().GetNodes();
		//auto inference_node = dynamic_cast<InferenceNode*>(m_p_graph->GetNodesByTypeAll("Inference")[0]);
		//auto batch_size = inference_node->get_m_param().get_m_batch_size();

		for (auto it = nodes->begin(); it != nodes->end(); it++) {
			string type = (*it)->GetType();
			if (type == "Image" || type == "BatchSplit") {
				continue;
			}
			//else if (type == "BatchMerge") {
			//	BatchMergeNode* p_node = dynamic_cast<BatchMergeNode*>((*it).get());
			//	BatchMergeParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "GpuSchedule") {
			//	GpuScheduleNode* p_node = dynamic_cast<GpuScheduleNode*>((*it).get());
			//	GpuScheduleParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "MemoryCopy") {
			//	MemoryCopyNode* p_node = dynamic_cast<MemoryCopyNode*>((*it).get());
			//	MemoryCopyParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "Inference") {
			//	InferenceNode* p_node = dynamic_cast<InferenceNode*>((*it).get());
			//	InferenceParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "PostProcess") {
			//	PostProcessNode* p_node = dynamic_cast<PostProcessNode*>((*it).get());
			//	PostProcessParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}

			if (nodes_optimized.size() > 0) {
				shared_ptr<Node> p_node = (*it);
				p_node->GetBottom()[0] = nodes_optimized[nodes_optimized.size() - 1]->GetName();
			}

			nodes_optimized.push_back(shared_ptr<ScheduleNode>(dynamic_cast<ScheduleNode*>(it->get())));
		}
	}

	void OfflineOptimizer::Optimize() {
		auto& nodes_optimized = m_p_graph->get_m_nodes();
		nodes_optimized.clear();
		auto nodes = m_p_graph->get_m_conf().GetNodes();
		//auto inference_node = dynamic_cast<InferenceNode*>(m_p_graph->GetNodesByTypeAll("Inference")[0]);
		//auto batch_size = inference_node->get_m_param().get_m_batch_size();

		for (auto it = nodes->begin(); it != nodes->end(); it++) {
			string type = (*it)->GetType();
			if (type == "Image" || type == "BatchMerge") {
				continue;
			}
			//else if (type == "BatchSplit") {
			//	BatchSplitNode* p_node = dynamic_cast<BatchSplitNode*>((*it).get());
			//	BatchSplitParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "GpuSchedule") {
			//	GpuScheduleNode* p_node = dynamic_cast<GpuScheduleNode*>((*it).get());
			//	GpuScheduleParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "MemoryCopy") {
			//	MemoryCopyNode* p_node = dynamic_cast<MemoryCopyNode*>((*it).get());
			//	MemoryCopyParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "Inference") {
			//	InferenceNode* p_node = dynamic_cast<InferenceNode*>((*it).get());
			//	InferenceParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}
			//else if (type == "PostProcess") {
			//	PostProcessNode* p_node = dynamic_cast<PostProcessNode*>((*it).get());
			//	PostProcessParam& param = p_node->get_m_param();
			//	param.set_m_batch_size(batch_size);
			//}

			if (nodes_optimized.size() > 0) {
				shared_ptr<Node> p_node = (*it);
				p_node->GetBottom()[0] = nodes_optimized[nodes_optimized.size() - 1]->GetName();
			}

			nodes_optimized.push_back(shared_ptr<ScheduleNode>(dynamic_cast<ScheduleNode*>(it->get())));
		}
	}

}
