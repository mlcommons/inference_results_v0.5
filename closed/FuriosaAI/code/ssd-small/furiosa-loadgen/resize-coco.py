import logging
import sys
import time
import os

import cv2
import numpy as np


def maybe_resize(img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
        # some images might be grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
        im_height, im_width, _ = dims
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    return img

def pre_process_mobilenet(img, dims=None, need_transpose=False):
    img = maybe_resize(img, dims)
    img = np.asarray(img, dtype=np.uint8)
    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return np.asarray(img, dtype=np.uint8)

imageloc = sys.argv[1]
outloc = sys.argv[2]
for filename in os.listdir(imageloc):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(imageloc, filename))
        preprocessed = pre_process_mobilenet(image, [300, 300, 3])
        np.save(os.path.join(outloc, filename+".npy"), preprocessed)
