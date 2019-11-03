#ifndef __CV_IMAGE_UTILS__H
#define __CV_IMAGE_UTILS__H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>


/**
 * @brief Loads an image from a file and scales it to the specified size.
 * @param path[in] Image file name.
 * @param reqsize[in] Resulting image size (width and height).
 * @param mean[in] Mean values to be subtracted from each channel.
 * @param std_dev[in] Scale values to be divided by each channel.
 * @param imageSize[out] Pointer of location to store size, in bytes of loaded image.
 * @param invert_rgb[in] If true, red and blue channels are swapped, otherwise,
 * no change is done to the color channels.
 * @return Pointer to the buffer containing the image in CHW format.
 * @details \p mean and \p std_dev are applied to the image after all other
 * operations have been applied (resize, invert_rgb, etc.).
 */
std::vector<float> LoadImageOpenCVPreProcessing(const std::string& path);
//std::unique_ptr<NetConf> net_conf_ = get_net_conf("resnet50");

#endif
