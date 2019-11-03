"""
tflite backend (https://github.com/tensorflow/tensorflow/lite)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper

import backend

import sys


class BackendTflite(backend.Backend):
    def __init__(self):
        super(BackendTflite, self).__init__()
        self.sess = None
        self.lock = Lock()


    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tflite-calibrate"

    def image_format(self):
        # tflite is always NHWC
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        self.sess = interpreter_wrapper.Interpreter(model_path=model_path)
        self.sess.allocate_tensors()
        # keep input/output name to index mapping
        self.input2index = {i["name"]: i["index"] for i in self.sess.get_input_details()}
        self.output2index = {i["name"]: i["index"] for i in self.sess.get_output_details()}
        # keep input/output names
        self.inputs = list(self.input2index.keys())
        self.outputs = list(self.output2index.keys())

        ### Calibration.
        self.sample_count = 0 
        self.calibrate = True
        self.calibration_data = []

        if self.calibrate:
            # Converting the MLPerf ResNet50 GraphDef from file.
            graph_def_file = "models/resnet50_v1.pb"
            self.converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, ["input_tensor"], ["ArgMax"])
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = self.converter.convert()
            open("resnet50_v1.fp32.tflite", "wb").write(tflite_model)

        return self

    def predict(self, feed):
        print('Calibration sample count: ' + str(self.sample_count))
        self.sample_count = self.sample_count + 1

        self.lock.acquire()
        # set inputs
        for k, v in self.input2index.items():
            self.sess.set_tensor(v, feed[k])

        self.sample_count = self.sample_count + 1 
        if self.sample_count > 0:
            self.calibration_data.append(feed[k])

        self.sess.invoke()
        # get results
        res = [self.sess.get_tensor(v) for _, v in self.output2index.items()]

        self.lock.release()


        if self.calibrate:
            if len(self.calibration_data) > 100:
                print('Calibrating with len(self.calibration_data) = ' + str(len(self.calibration_data)))
                # Lets try to calibrate with data we've collected.
                def representative_dataset_gen():
                    for sample in self.calibration_data:
                        yield [sample]
                self.converter.representative_dataset = representative_dataset_gen
                tflite_quant_model = self.converter.convert()
                open("resnet50_v1.calibrated.tflite", "wb").write(tflite_quant_model)
                print('Wrote calibrated model...')
                sys.exit(0)

        return res
