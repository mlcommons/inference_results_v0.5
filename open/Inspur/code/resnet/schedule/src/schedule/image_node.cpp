#include "../config/input_file.h"
#include "init_node.h"
#include "image_node.h"


namespace schedule {

	shared_ptr<ImageNode> ImageNodeParser::Run(Node node) {
		string line = m_pfile->ReadLine();
		if (line != "image_param {")
			throw "read image_param fail";
		ImageParam param = this->ParseParam();

		string _ = m_pfile->ReadLine();

		return make_shared<ImageNode>(node.GetName(), node.GetType(), node.GetBottom(), node.GetTop(), param);
	}

	ImageParam ImageNodeParser::ParseParam() {
		size_t total_count = 1000;
		size_t loadable_set_size = 100;

		map<string, size_t*> key_value;
		key_value["total_count:"] = &total_count;
		key_value["loadable_set_size:"] = &loadable_set_size;
		this->ParseLines<size_t>(key_value, "", true);

		string _ = m_pfile->ReadLine();

		return ImageParam(total_count, loadable_set_size);
	}

}