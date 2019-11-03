#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic pop

#include <algorithm>
#include <vector>
#include "cv_image_utils.h"
#include "net_config.h"

std::unique_ptr<NetConf> net_conf_ = get_net_conf("resnet50");

void ResizeWithAspect(cv::Mat* sample, cv::Mat* sample_resized) {
  auto scale = net_conf_->aspect_scale;
  auto new_height = static_cast<int>(100. * net_conf_->height / scale);
  auto new_width = static_cast<int>(100. * net_conf_->width / scale);
  auto inter_pol = net_conf_->net_name == "resnet50" ? cv::INTER_AREA: cv::INTER_LINEAR;
  if ((*sample).rows > (*sample).cols) {
    auto res = static_cast<int>((*sample).rows * new_width / (*sample).cols);
    cv::resize(*sample, *sample_resized, cv::Size(new_width, res), (0, 0), (0, 0), inter_pol);
  } else {
    auto res = static_cast<int>((*sample).cols * new_height / (*sample).rows);
    cv::resize(*sample, *sample_resized, cv::Size(res, new_height), (0, 0), (0, 0), inter_pol);
  }
}

// resize image using rescale

void ResizeWithRescale(cv::Mat* sample, cv::Mat* sample_rescale) {
  auto aspect = static_cast<float>((*sample).cols) / (*sample).rows;
  if (aspect > 1) {
    auto res = static_cast<int>(net_conf_->rescale_size * aspect);
    cv::resize(*sample, *sample_rescale, cv::Size(res, net_conf_->rescale_size));
  } else {
    auto res = static_cast<int>(net_conf_->rescale_size / aspect);
    cv::resize(*sample, *sample_rescale, cv::Size(net_conf_->rescale_size, res));
  }
}


void CenterCrop(cv::Mat* sample_resized, cv::Mat* sample_roi) {
  int x = (*sample_resized).cols;
  int y = (*sample_resized).rows;
  int startx = static_cast<int>(std::floor(x * 0.5 - net_conf_->width * 0.5));
  int starty = static_cast<int>(std::floor(y * 0.5 - net_conf_->height * 0.5));
  cv::Rect roi(startx, starty, net_conf_->width, net_conf_->height);
  // roi image
  (*sample_roi) = (*sample_resized)(roi);
}

std::vector<float> LoadImageOpenCVPreProcessing (const std::string& path) 
{
  // wrap and process image files.
  cv::Mat mean(net_conf_->width, net_conf_->height, CV_32FC3,
         cv::Scalar(net_conf_->mean_value[0], net_conf_->mean_value[1], net_conf_->mean_value[2]));
  cv::Mat scale(net_conf_->width, net_conf_->height, CV_32FC3,
                   cv::Scalar(net_conf_->scale, net_conf_->scale, net_conf_->scale));
#pragma omp parallel for
  unsigned long long inputSize_ = net_conf_->channels * net_conf_->height * net_conf_->width;
  //for (size_t i = 0; i < imgNames.size(); ++i) {
    std::vector<float> input_data(inputSize_);
    float* input_data_ptr = &input_data[0];
    
    cv::Mat img = cv::imread(path);
    // convert the input image to the input image format of the network.
    cv::Mat sample;
    if (img.channels() == 3 && net_conf_->channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && net_conf_->channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && net_conf_->channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && net_conf_->channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
      sample = img;

    cv::Mat sample_resized;
    cv::Mat sample_roi;
    if (net_conf_->preprocess_method == "ResizeWithAspect")
      ResizeWithAspect(&sample, &sample_resized);
    else
      ResizeWithRescale(&sample, &sample_resized);
    CenterCrop(&sample_resized, &sample_roi);

    cv::Mat sample_float;
    if (net_conf_->channels == 3) {
      sample_roi.convertTo(sample_float, CV_32FC3);
    } else sample_roi.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_subtract, sample_normalized;
    if (net_conf_->net_name=="resnet50") {
      cv::subtract(sample_float, mean, sample_subtract);
      cv::multiply(sample_subtract, scale, sample_normalized);
    } else if (net_conf_->net_name=="mobilenetv1") {
      cv::subtract(sample_float, mean, sample_subtract);
      cv::divide(sample_subtract, mean, sample_normalized);
    }
    vector<cv::Mat> input_channels;
    std::string dataOrder_="NCHW";
    if (dataOrder_ == "NCHW") {
      for (auto j = 0; j < net_conf_->channels; ++j) {
        cv::Mat channel(net_conf_->height, net_conf_->width, CV_32FC1, input_data_ptr);
        input_channels.push_back(channel);
        input_data_ptr += net_conf_->width * net_conf_->height;
      }
      if (net_conf_->bgr2rgb) std::reverse(input_channels.begin(), input_channels.end());
      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      cv::split(sample_normalized, input_channels);
    } else if (dataOrder_ == "NHWC") {
      cv::Mat channel(net_conf_->height, net_conf_->width, CV_32FC3, input_data_ptr);
      sample_normalized.copyTo(channel);
    }
    return std::move(input_data);
  //}
}
