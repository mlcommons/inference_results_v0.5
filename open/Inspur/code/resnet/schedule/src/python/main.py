"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
# import array
# import collections
import json
import logging
import os
import sys
# import threading
import time
# from queue import Queue

import mlperf_schedule as sch
import numpy as np

import dataset
import imagenet
import coco

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet":
        (imagenet.Imagenet, dataset.pre_process_vgg, dataset.PostProcessCommon(offset=1),
         {"image_size": [224, 224, 3]}),
    "imagenet_mobilenet":
        (imagenet.Imagenet, dataset.pre_process_mobilenet, dataset.PostProcessCommon(offset=-1),
         {"image_size": [224, 224, 3]}),
    "coco-300":
        (coco.Coco, dataset.pre_process_coco_mobilenet, coco.PostProcessCoco(),
         {"image_size": [300, 300, 3]}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "backend": "tensorflow",
    },
    # resnet
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
        "model-name": "resnet50",
    },
    # mobilenet
    "mobilenet-tf": {
        "inputs": "input:0",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "dataset": "imagenet_mobilenet",
        "backend": "tensorflow",
        "model-name": "mobilenet",
    },
    # ssd-mobilenet
    "ssd-mobilenet-tf": {
        "inputs": "image_tensor:0",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "dataset": "coco-300",
        "backend": "tensorflow",
        "model-name": "ssd-mobilenet",
    },
}

SCENARIO_MAP = {
    "SingleStream": sch.TestScenario.SingleStream,
    "MultiStream": sch.TestScenario.MultiStream,
    "Server": sch.TestScenario.Server,
    "Offline": sch.TestScenario.Offline,
}

CASE_SETTINGS_MAP = {
    # resnet50-tf
    "resnet50-tf-SingleStream": {
        "schedule_config": "/root/wsh/schedule-benchmark/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/root/wsh/mlperf-data/preprocess",
        "dataset_path": "/root/wsh/mlperf-data/dataset-imagenet-ilsvrc2012-val",
        #"model_path": "/root/wsh/mlperf-trt-models-v6/resnet50_int8_bs1.trt",
        "model_path": "/root/wsh/mlperf-trt-models-v6/fp16/resnet50_fp16_ws2_bt256.trt",
        "model_name": "resnet50",
        "dataset": "imagenet",
        "profile": "resnet50-tf",
        "scenario": "SingleStream",
    },
    "resnet50-tf-MultiStream": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-imagenet-ilsvrc2012-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/resnet50_v1_16.trt",
        "model_name": "resnet50",
        "dataset": "imagenet",
        "profile": "resnet50-tf",
        "scenario": "MultiStream",
    },
    "resnet50-tf-Server": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-imagenet-ilsvrc2012-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/resnet50_v1_16.trt",
        "model_name": "resnet50",
        "dataset": "imagenet",
        "profile": "resnet50-tf",
        "scenario": "Server",
    },
    "resnet50-tf-Offline": {
        "schedule_config": "/root/wsh/schedule-benchmark/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/root/wsh/mlperf-data/preprocess",
        "dataset_path": "/root/wsh/mlperf-data/dataset-imagenet-ilsvrc2012-val",
        #"model_path": "/root/wsh/mlperf-trt-models-v6/resnet50_fp16_bs1024.trt",
        #"model_path": "/root/wsh/mlperf-trt-models-v6/fp16/resnet50_fp16_ws2_bt256.trt",
        "model_path": "/root/wsh/mlperf-trt-models-v6/fp16/resnet50_fp16_ws2_bt512.trt",
        "model_name": "resnet50",
        "dataset": "imagenet",
        "profile": "resnet50-tf",
        "scenario": "Offline",
    },

    # mobilenet-tf
    "mobilenet-tf-SingleStream": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-imagenet-ilsvrc2012-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/mobilenet_v1_1.0_224_1.trt",
        "model_name": "mobilenet",
        "dataset": "imagenet_mobilenet",
        "profile": "mobilenet-tf",
        "scenario": "SingleStream",
    },
    "mobilenet-tf-MultiStream": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-imagenet-ilsvrc2012-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/mobilenet_v1_1.0_224_16.trt",
        "model_name": "mobilenet",
        "dataset": "imagenet_mobilenet",
        "profile": "mobilenet-tf",
        "scenario": "MultiStream",
    },
    "mobilenet-tf-Server": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-imagenet-ilsvrc2012-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/mobilenet_v1_1.0_224_16.trt",
        "model_name": "mobilenet",
        "dataset": "imagenet_mobilenet",
        "profile": "mobilenet-tf",
        "scenario": "Server",
    },
    "mobilenet-tf-Offline": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-imagenet-ilsvrc2012-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/mobilenet_v1_1.0_224_16.trt",
        "model_name": "mobilenet",
        "dataset": "imagenet_mobilenet",
        "profile": "mobilenet-tf",
        "scenario": "Offline",
    },

    # ssd-mobilenet-tf
    "ssd-mobilenet-tf-SingleStream": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-coco-2017-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/ssd_mobilenet_1.trt",
        "model_name": "ssd-mobilenet",
        "dataset": "coco-300",
        "profile": "ssd-mobilenet-tf",
        "scenario": "SingleStream",
    },
    "ssd-mobilenet-tf-MultiStream": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-coco-2017-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/ssd_mobilenet_16.trt",
        "model_name": "ssd-mobilenet",
        "dataset": "coco-300",
        "profile": "ssd-mobilenet-tf",
        "scenario": "MultiStream",
    },
    "ssd-mobilenet-tf-Server": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-coco-2017-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/ssd_mobilenet_16.trt",
        "model_name": "ssd-mobilenet",
        "dataset": "coco-300",
        "profile": "ssd-mobilenet-tf",
        "scenario": "Server",
    },
    "ssd-mobilenet-tf-Offline": {
        "schedule_config": "/root/schedule/mlperf_inference_schedule.prototxt",
        "config": "mlperf.conf",
        "backend": "trt",
        "data_format": "NCHW",
        "cache_path": "/data/dataset-imagenet-ilsvrc2012-valILSVRC2012_img_val/preprocess",
        "dataset_path": "/data/dataset-coco-2017-val",
        "model_path": "/media/sunhy/inference-master-test/inference-master/v0.5/classification_and_detection/engineer/ssd_mobilenet_16.trt",
        "model_name": "ssd-mobilenet",
        "dataset": "coco-300",
        "profile": "ssd-mobilenet-tf",
        "scenario": "Offline",
    },
}


last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-case", choices=CASE_SETTINGS_MAP.keys(), help="test case")
    parser.add_argument("--schedule-config", help="test case")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", help="path to the dataset")
    parser.add_argument("--dataset-list", help="path to the dataset list")
    parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--model-name", help="name of the mlperf model, ie. resnet50")
    parser.add_argument("--model-path", help="model file")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs")
    parser.add_argument("--outputs", help="model outputs")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument("--cache-path", default="", help="cache path")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--config", default="../mlperf.conf", help="mlperf rules config")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples-per-query", type=int, help="mlperf multi-stream sample per query")
    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100. * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
        name, result["qps"], result["mean"], took, acc_str,
        len(result_list), buckets_str))


def main():
    global last_timeing
    args = get_args()

    if args.test_case:
        for key in CASE_SETTINGS_MAP[args.test_case]:
            value = CASE_SETTINGS_MAP[args.test_case][key]
            if key == "model_path" and args.max_batchsize:
                import re
                to_be_replaced = re.compile("\d+\.trt")
                value = to_be_replaced.sub(str(args.max_batchsize) + ".trt", value)
                print("new model path: ", value)
            setattr(args, key, value)

    log.info(args)

    config = os.path.abspath(args.config)
    if not os.path.exists(config):
        log.error("{} not found".format(config))
        sys.exit(1)

    # override image format if given
    image_format = args.data_format
    if not image_format:
        log.error("image_format invalid: {}".format(image_format))
        sys.exit(1)

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count
    # if count:
    #     count_override = True

    # dataset to use
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = wanted_dataset(data_path=args.dataset_path,
                        image_list=args.dataset_list,
                        name=args.dataset,
                        image_format=image_format,
                        pre_process=pre_proc,
                        use_cache=args.cache,
                        cache_dir=args.cache_path,
                        count=count,
                        **kwargs)

    final_results = {
        "runtime": "TensorRT",
        "version": "5.1.2",
        "time": int(time.time()),
        "cmdline": str(args),
    }

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    #
    # make one pass over the dataset to validate accuracy
    #
    # count = ds.get_item_count()

    scenario = SCENARIO_MAP[args.scenario]

    def process_latencies(latencies_ns):
        # called by loadgen to show us the recorded latencies
        global last_timeing
        last_timeing = [t / NANO_SEC for t in latencies_ns]

    settings = sch.GetInferenceSettings()
    settings.FromConfig(config, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = sch.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = sch.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = sch.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    # if count_override:
    #     settings.min_query_count = count
    #     settings.max_query_count = count

    if scenario == 'Offline':
        settings.min_query_count = 1
        settings.max_query_count = 1

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)

    settings.qsl_rng_seed = 0x2b7e151628aed2a6
    settings.sample_index_rng_seed = 0x093c467e37db0c7a
    settings.schedule_rng_seed = 0x3243f6a8885a308d
    
    ds_label = []
    if args.dataset=='coco-300' or args.dataset=='coco-1200-tf':
        for item in ds.label_list:
            ds_label.append(item[0])
    else:
        ds_label = ds.label_list

    if not os.path.exists(args.schedule_config):
        log.error("schedule config path not exist: {}".format(args.schedule_config))
        sys.exit(1)

    if not os.path.exists(args.dataset_path):
        log.error("dataset path not exist: {}".format(args.dataset_path))
        sys.exit(1)

    if not os.path.exists(args.model_path):
        log.error("cache dir not exist: {}".format(args.model_path))
        sys.exit(1)

    if not os.path.exists(ds.cache_dir):
        log.error("cache dir not exist: {}".format(ds.cache_dir))
        sys.exit(1)

    sch.InitSchedule(args.schedule_config,
                     settings, args.dataset, args.dataset_path, ds.cache_dir, args.model_path, args.profile, args.backend,
                     args.accuracy,
                     [SUPPORTED_PROFILES[args.profile]["inputs"] if "inputs" in SUPPORTED_PROFILES[args.profile] else ""],
                     [SUPPORTED_PROFILES[args.profile]["outputs"] if "outputs" in SUPPORTED_PROFILES[args.profile] else ""],
                     ds.image_list,
                     ds_label)
    sch.InitMLPerf(process_latencies)

    log.info("starting {}".format(scenario))

    sch.StartTest()

    upload_results = sch.UploadResults()
    post_proc.update_results(upload_results)
 
    if args.dataset.startswith("coco"):
        results_coco = []
        upload_results_data = sch.UploadResultsCoco()
        for batch in upload_results_data:
            batch_detects = []
            for image in batch:
                batch_detects.extend(image)
            results_coco.append(batch_detects)
        post_proc.update_results_coco(results_coco)

    result_dict = {"good": 0, "total": 0, "scenario": str(scenario)}
  
    if args.accuracy:
        post_proc.finalize(result_dict, ds, output_dir=args.output)
        last_timeing.append(0.0)
    else:
        result_dict["good"] = post_proc.good
        result_dict["total"] = post_proc.total
    
    print(result_dict)

    add_results(final_results, "{}".format(scenario),
               result_dict, last_timeing, time.time() - sch.GetLastLoad(), args.accuracy)
   # print(last_timeing)
    #
    # write final results
    #
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)
    #print(final_results)

if __name__ == "__main__":
    main()
