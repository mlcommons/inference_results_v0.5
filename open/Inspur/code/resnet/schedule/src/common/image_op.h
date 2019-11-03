#ifndef __IMAGE_OP_H__
#define __IMAGE_OP_H__


namespace common {

	// HWC -> CHW
	template <typename Stype, typename Dtype>
	inline void hwc2chw(size_t ch, size_t w, size_t h, const Stype* src, Dtype* dst) {
		size_t index = 0UL;
		const size_t hw_stride = w * h;
		for (size_t s = 0UL; s < hw_stride; ++s) {
			size_t stride_index = s;
			for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
				dst[stride_index] = static_cast<Dtype>(src[index++]);
			}
		}
	}

	// CHW -> HWC
	template <typename Stype, typename Dtype>
	inline void chw2hwc(size_t ch, size_t w, size_t h, const Stype* src, Dtype* dst) {
		size_t index = 0UL;
		const size_t hw_stride = w * h;
		for (size_t s = 0UL; s < hw_stride; ++s) {
			size_t stride_index = s;
			for (size_t c = 0UL; c < ch; ++c, stride_index += hw_stride) {
				dst[index++] = static_cast<Dtype>(src[stride_index]);
			}
		}
	}

}
#endif // !__IMAGE_OP_H__
