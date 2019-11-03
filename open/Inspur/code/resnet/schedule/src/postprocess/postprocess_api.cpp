//#include "postproc_coco.h"
//#include "postproc_argmax.h"
//#include "postproc_common.h"
//#include "postprocess_base.h"
//#include <pybind11/pybind11.h>
//#include <pybind11/functional.h>
//#include <pybind11/stl.h>
//#include <functional>
//
//
//
////这里为了局部编译通过临时做一下对象实例化，整体联调的时候把实现实例化的文件include到这里应该就可以
//int64_t offset_;
//bool use_inv_map0_;
//float threshold_;
//postprocess::PostProcessBase<float>* PostProc_Common = new postprocess::PostProcCommon<float>(offset_);
//postprocess::PostProcessBase<float>* PostProc_ArgMax = new postprocess::PostProcArgMax<float>(offset_);
//postprocess::PostProcessBase<float>* PostProc_Coco = new postprocess::PostProcessCoco<float>();
//postprocess::PostProcessBase<float>* PostProc_CocoPt = new postprocess::PostProcessCocoPt<float>(use_inv_map0_, threshold_);
//postprocess::PostProcessBase<float>* PostProc_CocoOnnx = new postprocess::PostProcessCocoOnnx<float>();
//postprocess::PostProcessBase<float>* PostProc_CocoTf = new postprocess::PostProcessCocoTf<float>();
////------------------------------------------------
//
//
//
//namespace postprocess {
//	namespace py = pybind11;
//	
//	std::map<std::string, size_t> UploadResult_Common() {
//		return PostProc_Common->uploadresult();
//	}
//		
//	std::map<std::string, size_t> UploadResult_Argmax() {
//		return PostProc_ArgMax->uploadresult();
//	}
//
//	std::map<std::string, size_t> UploadResult_Coco() {
//		return PostProc_Coco->uploadresult();
//	}
//
//	std::map<std::string, size_t> UploadResult_CocoPt() {
//		return PostProc_CocoPt->uploadresult();
//	}
//
//	std::map<std::string, size_t> UploadResult_CocoOnnx() {
//		return PostProc_CocoOnnx->uploadresult();
//	}
//
//	std::map<std::string, size_t> UploadResult_CocoTf() {
//		return PostProc_CocoTf->uploadresult();
//	}
//
//	std::vector<std::vector<std::vector<std::vector<float>>>> UploadCocoMatrix_() {
//		return PostProc_Coco->uploadcocoresult();
//	}
//
//	std::vector<std::vector<std::vector<std::vector<float>>>> UploadCocoMatrix_Pt() {
//		return PostProc_CocoPt->uploadcocoresult();
//	}
//
//	std::vector<std::vector<std::vector<std::vector<float>>>> UploadCocoMatrix_Onnx() {
//		return PostProc_CocoOnnx->uploadcocoresult();
//	}
//
//	std::vector<std::vector<std::vector<std::vector<float>>>> UploadCocoMatrix_Tf() {
//		return PostProc_CocoTf->uploadcocoresult();
//	}
//
//	
//	PYBIND11_MODULE(postproc, m) {
//
//		m.def("UploadResult_Common", &UploadResult_Common);
//
//		m.def("UploadResult_Argmax", &UploadResult_Argmax);
//		
//		m.def("UploadResult_Coco", &UploadResult_Coco);
//
//		m.def("UploadResult_CocoPt", &UploadResult_CocoPt);
//
//		m.def("UploadResult_CocoOnnx", &UploadResult_CocoOnnx);
//
//		m.def("UploadResult_CocoTf", &UploadResult_CocoTf);
//
//		m.def("UploadCocoMatrix_", &UploadCocoMatrix_);
//
//		m.def("UploadCocoMatrix_Pt", &UploadCocoMatrix_Pt);
//
//		m.def("UploadCocoMatrix_Onnx", &UploadCocoMatrix_Onnx);
//
//		m.def("UploadCocoMatrix_Tf", &UploadCocoMatrix_Tf);
//
//	}
//}
//
