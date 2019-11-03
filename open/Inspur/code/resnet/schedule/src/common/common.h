#ifndef __COMMON_H__
#define __COMMON_H__


namespace common {

	//prototxt
	enum class ConfigStage {
		COMMON = 0,
		NODE
	};

	//node
	enum class NodeType
	{
		IMAGE_DATA = 0,
		CONVOLUTION,
		POOLING,
		BATCH_NORM,
		RELU,
		INNER_PRODUCT,
		ELTWISE,
		SOFTMAX,
		ACCURACY,
	};

	//WeightFillerType
	enum class WeightFillerType {
		MSRA = 0
	};

	//BiasFillerType
	enum class BiasFillerType {
		CONSTANT = 0
	};

	enum class StringStripType {
		LEFTSTRIP = 0,
		RIGHTSTRIP,
		BOTHSTRIP
	};

}
#endif // !__COMMON_H__
