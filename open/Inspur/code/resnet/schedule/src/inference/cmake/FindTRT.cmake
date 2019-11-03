# Find the TensorRT library
#
# The following variables are optionally searched for defaults
#  TRT_ROOT_DIR:    Base directory where all TRT components are found
#
# The following are set after configuration is done:
#  TRT_FOUND
#  TRT_INCLUDE_DIR
#  TRT_LIBRARIES

find_path(TRT_INCLUDE_DIR NAMES NvInfer.h PATHS ${TRT_ROOT_DIR}/include)

set(_trt_libs nvinfer nvcaffe_parser nvparsers nvinfer_plugin)

foreach (_trt_lib ${_trt_libs})
  string(TOUPPER ${_trt_lib} _trt_lib_upper)
  find_library(${_trt_lib_upper}_LIBRARY NAMES ${_trt_lib} PATHS ${TRT_ROOT_DIR}/lib)
  mark_as_advanced(${_trt_lib_upper}_LIBRARY)
  list(APPEND TRT_LIBRARIES ${${_trt_lib_upper}_LIBRARY})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TRT DEFAULT_MSG TRT_INCLUDE_DIR TRT_LIBRARIES)

if(TRT_FOUND)
  message(STATUS "Found TensorRT (include: ${TRT_INCLUDE_DIR}, libraries: ${TRT_LIBRARIES})")
  mark_as_advanced(TRT_INCLUDE_DIR TRT_LIBRARIES)
endif()
