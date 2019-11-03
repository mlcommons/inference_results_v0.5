#pragma once
#include "../common/macro.h"
#include "../config/node.h"
#include "../config/parser.h"
#include "inner_thread.h"
#include "data_set.h"
#include "blocking_queue.h"
#include "schedule.h"

using namespace std;


namespace schedule {

	class ImageParam {
		CREATE_SIMPLE_ATTR_SET_GET(m_total_count, size_t)
		CREATE_SIMPLE_ATTR_SET_GET(m_loadable_set_size, size_t)

	public:
		void Construct(ImageParam& param) {
			m_total_count = param.m_total_count;
			m_loadable_set_size = param.m_loadable_set_size;
		}
		ImageParam(size_t total_count, size_t loadable_set_size) {
			m_total_count = total_count;
			m_loadable_set_size = loadable_set_size;
		};
		ImageParam(ImageParam& param) {
			Construct(param);
		};
		ImageParam(ImageParam&& param) noexcept {
			Construct(param);
		};
		ImageParam& operator=(ImageParam&& param) noexcept {
			Construct(param);
			return *this;
		};
		virtual ~ImageParam() {};
	};

	class DataSet;
	class ImageNode : public ScheduleNode {
		CREATE_SIMPLE_ATTR_SET_GET(m_param, ImageParam)
		CREATE_SIMPLE_ATTR_GET(m_data_set, DataSet)

	public:
		ImageNode(string name, string type, vector<string> bottom, vector<string> top,
			ImageParam param)
			: ScheduleNode(name, type, bottom, top),
			m_param(param) {};
		~ImageNode() {};
	};

	class InputFile;
	class ImageNodeParser : public NodeParser {
	public:
		ImageNodeParser(InputFile* pfile, Config* pconf) : NodeParser(pfile, pconf) {};
		~ImageNodeParser() {};

		shared_ptr<ImageNode> Run(Node node);
		ImageParam ParseParam();
	};

}
