#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

import tensorflow as tf
import numpy as np

from collections import Counter
from multiprocessing import Queue, Process, Value, get_logger
from time import sleep
from ctypes import c_bool

import backend

from hailo_platform.common.jlf import readjlf
from hailo_platform.drivers.hw_object import HailoUdpControllerObject
from hailo_platform.drivers.hailo_controller.hailo_controller import HailoBroadcastController

from hailo_sdk_client import ClientRunner

# from hailo_ml_utils.ssd_utils.ssd_postproc import add_ssd_postproc

from demos_backbone.demo_wrapper import DemoWrapper
from demos_ui.control import UIControl

IDLE_POWER_FACTOR = 0.04

def _get_jlfs_paths(model):
    my_dir = os.path.dirname(__file__)

    if not isinstance(model, str):
        raise TypeError('model must be a string not {}'.format(model.__class__.__name__))

    return os.path.join('models_files', model, 'results', 'jlfs', '*.jlf')

    if model.lower() in ['resnet50', 'resnet_v1_50']:
        jlfs_paths = os.path.join(my_dir, 'models_files', 'resnet_v1_50', 'jlfs', '*')
    elif model.lower() in ['mobilenet', 'mobilenet_v1']:
        jlfs_paths = os.path.join(my_dir, 'models_files', 'mobilenet_v1', 'jlfs', '*')
        jlfs_paths = os.path.join(my_dir, 'trial', '*')
    elif model.lower() in ['mobilenet_ssd_nms', 'mobilenet_ssd', 'ssd']:
        jlfs_paths = os.path.join(my_dir, 'models_files', 'mobilenet_ssd_nms', 'jlfs', '*')
        #  jlfs_paths = os.path.join(my_dir, 'models_files', 'mobilenet_ssd_nms.new', 'jlfs', '*')
        #  jlfs_paths = os.path.join(my_dir, 'models_files', 'mobilenet_ssd_nms.old', 'jlfs', '*')
        #  jlfs_paths = os.path.join(my_dir, 'models_files', 'mobilenet_ssd_nms_ibc_eq_8bit_bias_16img_90_classes', 'jlfs', '*')
        #  jlfs_paths = os.path.join(my_dir, 'auto', '*')
    else:
        raise ValueError(f'Unsupported model: {model}')

    return jlfs_paths

def get_jlfs(model):
    from glob import glob
    jlfs_paths = _get_jlfs_paths(model)
    jlfs = []

    for jlfp in glob(jlfs_paths):
        with open(jlfp, 'rb') as jlf_file:
            jlfs.append(jlf_file.read())
    print(jlfs_paths)
    return jlfs


def batch_nms_post_processing(net_output,
                              score_threshold=0.3,
                              #max_proposals=100,
                             ):
    '''
    process batched nms output (post-infer).
    
    Args:
    net_output(np.array) - the output of the post infer. should be of size(batch, classes, 5, max_proposals)
    score_threshold(float) - the score threshold by which to select boxes.
    
    Returns:
    boxes(list) - a list of length batches. each element in the list is a np array of shape (detections,4)
    scores(list) - a list of length batches. each element in the list is a np array of shape (detections,)
    classes(list) - a list of length batches. each element in the list is a np array of shape (detections,)
    num_detection(np.array) - each element gives the #detections in each batch
    '''
    indices = np.argwhere(net_output[:, :, 4, :] > score_threshold)
    boxes = net_output[indices[:,0], indices[:,1] ,:4, indices[:,2]]
    scores = net_output[indices[:,0], indices[:,1] ,4, indices[:,2]]
    classes = indices[:,1] + 1
    batch = indices[:,0]
    d = Counter(batch)
    batch_size = net_output.shape[0]
    idx = np.zeros(shape=(batch_size-1),dtype=np.uint16)
    num_detections = np.zeros(shape=batch_size,dtype=np.uint16)
    last_ind = 0
    for i in range(batch_size-1):
        if i in d.keys():
            last_ind += d[i]
            num_detections[i] = d[i]
        else:
            pass
        idx[i] = last_ind
    num_detections[batch_size-1] = 0 if not batch_size-1 in d else d[batch_size-1]
    classes = np.split(classes, idx)
    boxes = np.split(boxes, idx)
    scores = np.split(scores, idx)
    return num_detections, boxes, scores, classes

    indices = np.argwhere(net_output[:, :, 4, :] > score_threshold)
    boxes = net_output[indices[:,0], indices[:,1] ,:4, indices[:,2]]
    scores = net_output[indices[:,0], indices[:,1] ,4, indices[:,2]]
    classes = indices[:,1] + 1
    batch = indices[:,0]
    # split the array by batch
    idx = np.where(batch[1:] != batch[:-1])[0] + 1
    classes = np.split(classes, idx)
    boxes = np.split(boxes, idx)
    scores = np.split(scores, idx)
    idx = np.append(0, np.append(idx, batch.size))
    num_detections = idx[1:]-idx[:-1]

    return num_detections, boxes, scores, classes


def power_measurement_process(target, power_measurement_queue, control):
    logger = get_logger()
    logger.info('power process')

    avg = 0
    counted = 0
    first_measure = None

    with DemoWrapper._DemoPowerMeasurement(target, 0) as pm_tool:
        while not control.is_shut():
            sleep(0.1)

            while control.is_running():
                samples = pm_tool.get()

                if first_measure is None and samples:
                    first_measure = max(samples)

                samples = [sample for sample in samples if sample > (first_measure * (1 + IDLE_POWER_FACTOR))]

                # if sample > (first_measure * (1 + IDLE_POWER_FACTOR)):
                for sample in samples:
                    avg = (avg * counted + sample) // (counted + 1)
                    counted += 1
                    power_measurement_queue.put(avg)
                sleep(5)


class Parser(object):
    def __init__(self, network_db):
        self._db = network_db
        self._name = self._db['network_name']
        self._results_dir = os.path.join('models_files', self._name, 'results')
        self._ckpt_path = self._get_ckpt_path()

        self._create_results_dir()

    def set_shapes(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            meta_path = os.path.extsep.join((self._ckpt_path, 'meta'))
            new_saver = tf.train.import_meta_graph(meta_path)
            with tf.Session() as sess:
                new_saver.restore(sess, self._ckpt_path)
                image_tensor = tf.get_default_graph().get_tensor_by_name('Preprocessor/sub:0')

                shapes = [None, self._db['img_dims'][0], self._db['img_dims'][1], 3]
                image_tensor.set_shape(shapes)

                ops = tf.get_default_graph().get_operations()
                variable_names = [op.name + ':0' for op in ops if "Variable" in op.type]
                
                for name in variable_names:
                    tensor = tf.get_default_graph().get_tensor_by_name(name)
                    tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, tensor)
                saver = tf.train.Saver()
                new_ckpt_path = os.path.join(self._results_dir, 'model.ckpt')
                saver.save(sess, new_ckpt_path)
                self._ckpt_path = new_ckpt_path

    def parse(self):
        return self._parse(self._name, *self._db['nodes'])

    def dump(self, hn, npz):
        self._dump(self._name, hn, npz)

    @property
    def results_dir(self):
        return self._results_dir

    def _get_ckpt_path(self):
        ckpt_path = str.join(os.path.extsep, self._db['network_path'][0].split(os.path.extsep)[:-1])
        return ckpt_path

    def _create_results_dir(self):
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

    def _parse(self, net_name, start_node_name, end_node_names):
        """ Thin  wrapper over SDK's parsing API
        """
        runner = ClientRunner(hw_arch='hailo8')

        if not isinstance(end_node_names, list):
            end_node_names = [end_node_names]

        hn_data, npz_data = runner.translate_tf_model(
            self._ckpt_path, 
            net_name,
            start_node_name=start_node_name,
            end_node_names=end_node_names)

        return hn_data, npz_data

    def _dump(self, net_name, hn_data, npz_data):
        results_dir = os.path.join('models_files', net_name, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        with open(os.path.join(results_dir, f'{net_name}.hn'), 'w') as hn_file:
            hn_file.write(hn_data)

        np.savez(os.path.join(results_dir, f'{net_name}.npz'), **npz_data)


def erase_debug_from_jlfs(jlfs):
    params_data = jlfs[0]
    params_jlf = readjlf.open_jlf(params_data)
    try:
        bad_ent = next(ent for ent in params_jlf[1:] if ent.entry_type == 8)
        new = params_data[:len(params_data) - bad_ent.size]
    except StopIteration:
        new = params_data

    return (new,) + jlfs[1:]


class BackendHailo(backend.Backend):
    MODELS = {
        'resnet50': 'resnet_v1_50',
        'resnet_v1_50': 'resnet_v1_50',
        'mobilenet': 'mobilenet',
        'mobilenet_v1': 'mobilenet',
        'mobilenet_ssd': 'mobilenet_ssd_nms',
        'ssd': 'mobilenet_ssd_nms',
        }

    def __init__(self):
        super(BackendHailo, self).__init__()
        self.inputs.append('hailo')
        self._model = None
        self._translate_inputs = True

        # Setting the hw target
        self._board_ip = HailoBroadcastController().identify()[0].ip
        self._target = HailoUdpControllerObject(self._board_ip)
        self._context_manager = None
        self._control = UIControl()

        # Setting power measurement
        self._power_measurement_queue = Queue(maxsize=1000)
        self._measurements = []

    def version(self):
        return '2.4.1'

    def name(self):
        return 'hailo'

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return 'NHWC'

    def load(self, model, inputs=None, outputs=None):
        self._load_model(model)
        self._target.load_jlfs(get_jlfs(self._model))
        self._run_power_measure()
        self._target.set_debug()
        self._context_manager = self._target.use_device('throughput_optimized')
        self._context_manager.__enter__()

        # Run
        self._control.resume()

        return self

    def predict(self, feed):
        # inputs = [x for x in feed.values()]
        shaped_inputs = np.concatenate(list(feed.values()))
        # TODO: Find a way to deal with it
        # results = self._target.infer(shaped_inputs)
        results = self._target.infer(shaped_inputs, translate_input=self._translate_inputs)

        #  self._read_power_measure()

        if self._model == 'mobilenet':
            return results
        elif self._model == 'resnet_v1_50':
            return [[res.argmax() for res in results[0]]]
        elif self._model == 'mobilenet_ssd_nms':
            new_results = batch_nms_post_processing(results[0])
            #  new_results = batch_nms_post_processing(results[0][:,1:,:,:])
            return new_results

    def close(self):
        self._finalalize_power_measurements()
        self._close_contexts()
        sleep(0.5)
        self._close_a_process(self._power_measurement, self._power_measurement_queue)

    def _load_model(self, model):
        if model not in type(self).MODELS:
            raise ValueError(f'Unsupported model: {model}')

        self._model = type(self).MODELS[model]
        if self._model == 'mobilenet_ssd_nms':
            self._translate_inputs = False

    def _build(self):
        with open(os.path.join('models_files', 'networks.json'), 'r') as networks_file:
            network_db = json.load(networks_file)[self._model]

        parser = Parser(network_db)

        if self._model == 'mobilenet_ssd_nms':
            parser.set_shapes()
        hn, npz = parser.parse()
        parser.dump(hn, npz)

        net_name = network_db['network_name']
        input_hn = os.path.join(parser.results_dir, os.path.extsep.join((net_name, 'hn')))
        output_hn = input_hn
        input_npz = os.path.join(parser.results_dir, os.path.extsep.join((net_name, 'npz')))
        output_npz = input_npz

        if network_db['task'] == 'detection':
            config_json = os.path.join('models_files', self._model, 'bbox_info.json')
            nms = True
            nms_scores_th = network_db['score_threshold']
            nms_iou_th = network_db['nms_iou_thresh']
            max_prop_per_class = 10
            centers_scale_factor = 10
            scores_scale_factor = 5
            classes = network_db['classes'] + 1
            background_removal = True
            background_removal_index = 0

            add_ssd_postproc(input_hn, input_npz, output_hn, output_npz, config_json, nms,
                             nms_scores_th, nms_iou_th, max_prop_per_class, centers_scale_factor,
                             scores_scale_factor, classes, background_removal, background_removal_index)

        jlfs_dir = os.path.join(parser.results_dir, 'jlfs')
        if not os.path.exists(jlfs_dir):
            os.makedirs(jlfs_dir)

        jlfs = quantize(network_db, self._model, output_hn, output_npz, jlfs_dir)
        jlfs = erase_debug_from_jlfs(jlfs)

        for jlf_id, jlf_name in enumerate(['params.jlf', 'inference.jlf', 'meta_data.jlf', 'boot.jlf']):
            jlf_path = os.path.join(jlfs_dir, jlf_name)
            with open(jlf_path, 'wb') as jlf_file:
                jlf_file.write(jlfs[jlf_id])
    
    def _finalalize_power_measurements(self):
        tot = sum(self._measurements)
        if len(self._measurements):
            avg = tot / len(self._measurements)
            print(avg)

    def _read_power_measure(self):
        if not self._power_measurement_queue.empty():
            self._measurements.append(self._power_measurement_queue.get(False))

    def _run_power_measure(self):
        self._power_measurement = Process(
            target=power_measurement_process,
            args=(self._target, self._power_measurement_queue, self._control))
        self._power_measurement.start()

    def _close_a_process(self, process, queue):
        process.terminate()
        process.join()
        queue.close()

    def _close_contexts(self):
        for i in range(4):
            try:
                self._context_manager.__exit__(None, None, None)
                self._control.shutdown()
                sleep(1)
                self._control.close()
                break
            except:
                pass
