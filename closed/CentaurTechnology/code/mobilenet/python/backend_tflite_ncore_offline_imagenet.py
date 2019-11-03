"""
tflite-ncore backend (adapted from https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import os

from threading import Lock

import tensorflow as tf
print('Importing interpreter_wrapper...')
from tensorflow.lite.python import interpreter as interpreter_wrapper
print('Imported  interpreter_wrapper')

import numpy as np

import backend


class BackendTfliteNcoreOfflineImagenet(backend.Backend):
    def __init__(self):
        super(BackendTfliteNcoreOfflineImagenet, self).__init__()
        self.sess = None
        self.lock = Lock()

        self.sample_count = 0 # Debug
        self.batch_size = 128
        self.do_batches = True # True
        self.do_delegate = True # True

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tflite-ncore-offline-imagenet"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        self.sess = interpreter_wrapper.Interpreter(model_path=model_path)
        #self.sess = interpreter_wrapper.Interpreter(model_path=model_path, experimental_delegates=[delegate])

        if self.do_batches:
            input_details = self.sess.get_input_details()
            self.sess.resize_tensor_input(input_details[0]['index'], (self.batch_size, 224, 224, 3))

        # We have to load the delegate after resizing the input tensor for batches
        if self.do_delegate:
            print('Loading delegate... ' + os.getenv("NCORE_DELEGATE"))
            delegate = interpreter_wrapper.load_delegate(os.getenv("NCORE_DELEGATE"))
            self.sess.add_delegates(experimental_delegates=[delegate])

        self.sess.allocate_tensors()
        # keep input/output name to index mapping
        self.input2index = {i["name"]: i["index"] for i in self.sess.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.sess.get_output_details()}
        # keep input/output names
        self.inputs = list(self.input2index.keys())
        self.outputs = list(self.output2index.keys())
        return self

    def predict(self, feed):
        self.lock.acquire()
        batch_orig_size = -1
        # set inputs
        for k, v in self.input2index.items():
            if len(feed[k]) == self.batch_size or not self.do_batches:
                self.sess.set_tensor(v, feed[k])
            else:
                # Expand out trailing portion that is smaller than batch size
                batch_orig_size = len(feed[k])
                feedk_cpy = np.copy(feed[k])
                feedk_cpy.resize((self.batch_size, 224, 224, 3))
                self.sess.set_tensor(v, feedk_cpy)

        self.sess.invoke()

        # get results
        if batch_orig_size == -1:
            res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]
        else:
            # Truncate the null batch entries if we expanded out the input batches earlier
            res = [self.sess.get_tensor(v)[:batch_orig_size] for _, v in self.output2index.items()]

        self.lock.release()
        return res
