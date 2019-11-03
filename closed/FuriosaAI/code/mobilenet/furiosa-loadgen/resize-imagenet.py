import logging
import sys
import time
import os

import cv2
import numpy as np

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def pre_process_mobilenet(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return np.asarray(img, dtype=np.uint8)

imageloc = sys.argv[1]
outloc = sys.argv[2]
for filename in os.listdir(imageloc):
    if filename.endswith(".JPEG"):
        image = cv2.imread(os.path.join(imageloc, filename))
        preprocessed = pre_process_mobilenet(image, [224, 224, 3])
        #cv2.imwrite(os.path.join(outloc, filename), preprocessed)
        np.save(os.path.join(outloc, filename+".npy"), preprocessed)

