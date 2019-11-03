"""implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

import numpy as np
import tensorflow as tf

import dataloader
import ssd_constants


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Dataset(object):

  def __init__(self):
    self.arrival = None
    self.image_list = []
    self.label_list = []
    self.image_list_inmemory = []
    self.source_id_list_inmemory = []
    self.raw_shape_list_inmemory = []
    self.image_map = {}
    self.last_loaded = -1

  def preprocess(self, use_cache=True):
    raise NotImplementedError("Dataset:preprocess")

  def get_item_count(self):
    return len(self.image_list)

  def get_list(self):
    raise NotImplementedError("Dataset:get_list")

  def load_query_samples(self, sample_list):
    self.image_list_inmemory = []
    self.image_map = {}
    for i, sample in enumerate(sample_list):
      self.image_map[sample] = i
      img, _ = self.get_item(sample)
      self.image_list_inmemory.append(img)
    self.last_loaded = time.time()

  def unload_query_samples(self, sample_list):
    if sample_list:
      for sample in sample_list:
        del self.image_map[sample]
    else:
      self.image_map = {}
    self.image_list_inmemory = []

  def get_samples(self, item_list):
    data = [
        self.image_list_inmemory[self.image_map[item]] for item in item_list
    ]
    data = np.array(data)
    source_id = [
        self.label_list[self.idx_map[item]][ssd_constants.IDX]
        for item in item_list
    ]
    raw_shape = [
        self.label_list[self.idx_map[item]][ssd_constants.RAW_SHAPE]
        for item in item_list
    ]
    raw_shape = np.array(raw_shape)
    return (data, source_id, raw_shape), self.label_list[item_list]

  def get_image_list_inmemory(self):
    return np.array(self.image_list_inmemory)

  def get_indices(self, item_list):
    data = [self.image_map[item] for item in item_list]
    source_id = [
        self.label_list[self.idx_map[item]][ssd_constants.IDX]
        for item in item_list
    ]
    raw_shape = [
        self.label_list[self.idx_map[item]][ssd_constants.RAW_SHAPE]
        for item in item_list
    ]
    raw_shape = np.array(raw_shape)
    return (data, source_id, raw_shape), self.label_list[item_list]

  def get_item_loc(self, index):
    raise NotImplementedError("Dataset:get_item_loc")


class COCODataset(Dataset):

  def __init__(self,
               data_path,
               image_list,
               name,
               use_cache=0,
               image_format="NCHW",
               count=None,
               cache_dir=None,
               annotation_file=None,
               use_space_to_depth=False):
    super(COCODataset, self).__init__()
    if not cache_dir:
      cache_dir = os.getcwd()
    self.image_list = []
    self.label_list = []
    self.count = count
    self.use_cache = use_cache
    self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
    self.data_path = data_path
    self.space_to_depth = use_space_to_depth
    # input images are in HWC
    self.need_transpose = True if image_format == "NCHW" else False
    self.annotation_file = annotation_file

    not_found = 0
    tf.gfile.MakeDirs(self.cache_dir)

    start = time.time()
    ssd_dataloader = dataloader.SSDInputReader(
        data_path,
        transpose_input=self.need_transpose,
        space_to_depth=self.space_to_depth)
    dataset = ssd_dataloader()

    self.images = {}
    self.idx_map = {}
    with tf.gfile.Open(self.annotation_file, "r") as f:
      coco = json.load(f)
    for idx, i in enumerate(coco["images"]):
      self.images[i["id"]] = {
          "file_name": i["file_name"],
          ssd_constants.IDX: idx
      }

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    item = 0
    with tf.Session() as sess:
      sess.run(iterator.initializer)
      while True:
        try:
          labels = sess.run(next_element)
        except tf.errors.OutOfRangeError:
          break
        labels[ssd_constants.IDX] = self.images[labels[
            ssd_constants.SOURCE_ID]][ssd_constants.IDX]
        image_name = labels[ssd_constants.SOURCE_ID]
        dst = os.path.join(self.cache_dir, str(image_name) + ".npy")
        with tf.gfile.Open(dst, "wb") as fout:
          np.save(fout, labels[ssd_constants.IMAGE])
        labels.pop(ssd_constants.IMAGE)
        self.image_list.append(str(image_name))
        self.label_list.append(labels)
        self.idx_map[labels[ssd_constants.IDX]] = item
        item = item + 1

        # limit the dataset if requested
        if self.count and len(self.image_list) >= self.count:
          break

    time_taken = time.time() - start
    if not self.image_list:
      log.error("no images in image list found")
      raise ValueError("no images in image list found")
    if not_found > 0:
      log.info("reduced image list, %d images not found", not_found)

    log.info("loaded {} images, cache={}, took={:.1f}sec".format(
        len(self.image_list), use_cache, time_taken))

    self.label_list = np.array(self.label_list)

  def get_item(self, nr):
    """Get image by number in the list."""
    dst = os.path.join(self.cache_dir,
                       self.image_list[self.idx_map[nr]] + ".npy")
    with tf.gfile.Open(dst, "rb") as fout:
      img = np.load(fout)
    # TODO(wangtao): label is not useful for accuracy computation.
    # Fix the logic for accuracy mode.
    return img, 1

  def get_item_loc(self, nr):
    src = os.path.join(self.data_path, self.image_list[nr])
    return src
