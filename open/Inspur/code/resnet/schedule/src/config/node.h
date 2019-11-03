#ifndef __NODE_H__
#define __NODE_H__
#include <string>
#include <vector>

#include "../common/macro.h"

using namespace std;


namespace schedule {

	// For Convolution neural network.
	CREATE_SIMPLE_CLASS_GET_3(ImageDataParam, root_folder, string, source, string, batch_size, size_t)
	CREATE_SIMPLE_CLASS_GET_3(TransformParam, crop_size, size_t, mirror, bool, scale, float)
	CREATE_SIMPLE_CLASS_GET_1(WeightFiller, type, string)
	CREATE_SIMPLE_CLASS_GET_2(BiasFiller, type, string, value, float)
	CREATE_SIMPLE_CLASS_GET_1(Include, phase, string)
	CREATE_SIMPLE_CLASS_GET_6(ConvolutionParam, num_output, size_t, kernel_size, size_t,
		pad, size_t, stride, size_t, weight_filler, WeightFiller, bias_term, bool)
	CREATE_SIMPLE_CLASS_GET_3(PoolingParam, pool, string, kernel_size, size_t, stride, size_t)
	CREATE_SIMPLE_CLASS_GET_3(BatchNormParam, moving_average_fraction, float, eps, float, scale_bias, bool)
	CREATE_SIMPLE_CLASS_GET_3(InnerProductParam, num_output, size_t, weight_filler, WeightFiller, bias_filler, BiasFiller)
	CREATE_SIMPLE_CLASS_GET_1(EltwiseParam, operation, string)
	CREATE_SIMPLE_CLASS_GET_1(AccuracyParam, top_k, size_t)

	class Node {
	protected:
	public:
		Node(string name, string type, vector<string>& bottom, vector<string>& top) {
			m_name = name;
			m_type = type;
			m_bottom.swap(bottom);
			m_top.swap(top);
		};
		virtual ~Node() {};

		string GetName() { return m_name; };
		string GetType() { return m_type; };
		vector<string>& GetBottom() { return m_bottom; };
		vector<string>& GetTop() { return m_top; };

	private:
		string m_name;
		string m_type;
		vector<string> m_bottom;
		vector<string> m_top;
	};

	CREATE_NODE_CLASS_3(ImageDataNode, image_data_param, ImageDataParam,
		transform_param, TransformParam, include, Include)
	CREATE_NODE_CLASS_1(ConvolutionNode, param, ConvolutionParam)
	CREATE_NODE_CLASS_1(PoolingNode, param, PoolingParam)
	CREATE_NODE_CLASS_1(BatchNormNode, param, BatchNormParam)
	CREATE_NODE_CLASS_0(ReluNode)
	CREATE_NODE_CLASS_1(InnerProductNode, param, InnerProductParam)
	CREATE_NODE_CLASS_1(EltwiseNode, param, EltwiseParam)
	CREATE_NODE_CLASS_0(SoftmaxNode)
	CREATE_NODE_CLASS_2(AccuracyNode, param, AccuracyParam, include, Include)

}

#endif // !__NODE_H__