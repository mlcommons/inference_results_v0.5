"""
tflite-ncore backend (adapted from https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import os

from threading import Lock

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper

import backend


class BackendTfliteNcore(backend.Backend):
    def __init__(self):
        super(BackendTfliteNcore, self).__init__()
        self.sess = None
        self.lock = Lock()

        self.sample_count = 0 # Debug

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tflite-ncore"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        print('Loading delegate... ' + os.getenv("NCORE_DELEGATE"))
        # /n/scr_ncore/parvizp/Git/mlperf_inference/v0.5/classification_and_detection/ncore_py_delegate.so
        delegate = interpreter_wrapper.load_delegate(os.getenv("NCORE_DELEGATE"))
        self.sess = interpreter_wrapper.Interpreter(model_path=model_path, experimental_delegates=[delegate])
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
        # set inputs
        for k, v in self.input2index.items():
            self.sess.set_tensor(v, feed[k])
        self.sess.invoke()
        # get results
        res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]
        self.lock.release()
        return res
